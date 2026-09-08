"""The two tools the agent can reach for: SQL, and the compliance check.

**Why SQL exists at all.** Vector search retrieves; it does not compute. Asked
for ROAS across channels, a retrieval-only system finds chunks that *mention*
ROAS and does confident mental arithmetic on them, which is how you ship a fluent
wrong number into a board deck. Anything comparative, aggregate or arithmetic is
routed here, where the answer comes from the database or it does not come.

**How it is kept safe.** Not by asking the model nicely. The query is parsed and
rejected unless it is a single `SELECT`; a `LIMIT` is imposed; the connection is
opened read-only at the SQLite URI level, so a write is refused by the database
rather than by a prompt. Prompt instructions are the last line of defence here,
never the first.

**Why the generated SQL is shown to the user.** An analyst can check it in five
seconds, and a number nobody can check is a number nobody should paste into a
deck.
"""

import re
import sqlite3

from projects.marketingcopilot.database import DEMO_WORKSPACE, database, query

# The only table the tool may read, described the way the model needs it.
SCHEMA_PROMPT = """Table: campaigns
  name           TEXT     campaign name, e.g. 'Always-On Demand Gen'
  channel        TEXT     one of: linkedin, paid_search, email, events
  segment        TEXT     one of: fintech, insurance, banking
  quarter        TEXT     format '2024-Q3'
  spend          REAL     currency spent
  impressions    INTEGER
  clicks         INTEGER
  conversions    INTEGER
  pipeline_value REAL     pipeline generated

Derived measures you must compute, never assume a column for:
  CTR  = clicks * 1.0 / impressions
  CAC  = spend * 1.0 / conversions          (guard: conversions may be 0)
  ROAS = pipeline_value / spend
  CVR  = conversions * 1.0 / clicks

Rules:
  - SQLite dialect. One SELECT statement. No semicolon, no CTE chains, no writes.
  - Always filter workspace_id = 'demo'.
  - Round derived measures to 2 decimal places.
  - When comparing groups, GROUP BY and ORDER BY the measure.
"""

EXAMPLES = """Q: What was total spend on LinkedIn in Q3 2024?
A: SELECT SUM(spend) AS total_spend FROM campaigns WHERE workspace_id = 'demo' AND channel = 'linkedin' AND quarter = '2024-Q3'

Q: Which channel had the lowest cost per acquisition in Q3 2024?
A: SELECT channel, ROUND(SUM(spend) / NULLIF(SUM(conversions), 0), 2) AS cac FROM campaigns WHERE workspace_id = 'demo' AND quarter = '2024-Q3' GROUP BY channel ORDER BY cac ASC

Q: Compare fintech ROAS by channel this quarter
A: SELECT channel, ROUND(SUM(pipeline_value) / NULLIF(SUM(spend), 0), 2) AS roas FROM campaigns WHERE workspace_id = 'demo' AND segment = 'fintech' GROUP BY channel ORDER BY roas DESC
"""

MAX_ROWS = 50

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|"
    r"reindex|truncate|grant|revoke)\b", re.I)
_SELECT_START = re.compile(r"^\s*select\b", re.I)


class SQLError(Exception):
    pass


def validate_sql(statement: str) -> str:
    """Reject anything that is not a single, read-only SELECT.

    Runs before execution, not after. The checks are deliberately blunt: this is
    a guardrail, and a guardrail that is clever enough to be wrong is worse than
    one that is strict enough to be annoying.
    """
    cleaned = statement.strip().rstrip(";").strip()
    cleaned = re.sub(r"^```(?:sql)?|```$", "", cleaned, flags=re.I | re.M).strip()

    if not cleaned:
        raise SQLError("The model returned no query.")
    if not _SELECT_START.match(cleaned):
        raise SQLError("Only SELECT statements are allowed.")
    if ";" in cleaned:
        raise SQLError("Only a single statement is allowed.")
    if _FORBIDDEN.search(cleaned):
        raise SQLError("The query contains a write or schema operation.")
    if re.search(r"\bfrom\s+(?!campaigns\b)", cleaned, re.I):
        raise SQLError("This tool may only read the campaigns table.")
    if not re.search(r"\blimit\b", cleaned, re.I):
        cleaned = f"{cleaned} LIMIT {MAX_ROWS}"
    return cleaned


def run_sql(statement: str) -> list[dict]:
    """Execute against a read-only connection.

    `mode=ro` in the URI means SQLite itself refuses a write. That is the control
    that actually matters -- validation above can be outsmarted by a clever
    string, a read-only file descriptor cannot.
    """
    safe = validate_sql(statement)
    connection = sqlite3.connect(f"file:{database.path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(safe).fetchall()
        return [dict(row) for row in rows[:MAX_ROWS]]
    except sqlite3.Error as error:
        raise SQLError(f"The query failed: {error}") from error
    finally:
        connection.close()


def generate_sql(llm, question: str) -> str:
    prompt = (
        "You translate a marketing question into one SQLite SELECT statement.\n\n"
        f"{SCHEMA_PROMPT}\n{EXAMPLES}\n"
        "Return only the SQL. No explanation, no code fence.\n\n"
        f"Q: {question}\nA:")
    return llm.invoke(prompt).content.strip()


# ── compliance ───────────────────────────────────────────────────────────────

def load_rules(workspace_id: str = DEMO_WORKSPACE) -> list[dict]:
    """Every rule, always.

    Deliberately not a similarity search. There are five rules; retrieving the
    "top 3 most relevant" would mean the two it skipped are the two that catch
    the violation. A rulebook is one of the few things worth loading whole.
    """
    return query(
        "SELECT code, rule, severity, pattern FROM compliance_rules WHERE workspace_id = ?"
        " ORDER BY code", (workspace_id,))


def check_compliance(text: str, workspace_id: str = DEMO_WORKSPACE) -> dict:
    """Pattern-match the draft against every rule.

    Deterministic on purpose. An LLM judging its own output for compliance is
    both slower and less predictable than a regex, and for the class of
    violations that matter here -- absolute guarantees, unsourced superlatives,
    named customers -- the patterns are precise.

    This is a guardrail, not a gate. It catches the obvious violations before a
    human reviewer spends time on them; it cannot make anything legally safe,
    and the response says so.
    """
    findings = []
    for rule in load_rules(workspace_id):
        if not rule["pattern"]:
            continue
        match = re.search(rule["pattern"], text, re.I)
        if match:
            findings.append({
                "code": rule["code"],
                "severity": rule["severity"],
                "rule": rule["rule"],
                "matched": match.group(0),
            })
    return {
        "checked": True,
        "violations": findings,
        "passed": not findings,
        "note": ("Pattern-based pre-review. It reduces the volume reaching legal; "
                 "it is not legal approval."),
    }
