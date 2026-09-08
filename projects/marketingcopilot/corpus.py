"""The demo workspace: a marketing team's documents, numbers and rules.

A copilot with no corpus demonstrates nothing, so the project ships one. It is a
B2B SaaS company selling into financial services, which is chosen deliberately:
a regulated vertical is what makes the compliance step real rather than
decorative, and it makes "can we say this?" a question with an actual answer.

The three kinds of content mirror the three silos the project exists to join:

  documents  -> unstructured knowledge, goes to the vector store
  campaigns  -> numbers, goes to SQL and is never embedded
  rules      -> compliance, retrieved in full rather than by similarity

The campaign numbers are internally consistent -- clicks are a plausible share
of impressions, conversions of clicks -- because the copilot computes CTR, CAC
and ROAS from them and nonsense inputs would produce nonsense worth debugging.
"""

import hashlib
import json

from projects.marketingcopilot.database import (
    DEMO_WORKSPACE, database, new_id,
)

# ── campaign performance ─────────────────────────────────────────────────────
# The Q3 LinkedIn fintech row is the one the flagship question is about: heavy
# spend, poor conversion. Everything else is context that makes it look bad.

CAMPAIGNS = [
    # name, channel, segment, quarter, spend, impressions, clicks, conversions, pipeline
    ("Always-On Demand Gen", "linkedin", "fintech", "2024-Q2", 48000, 1_240_000, 9_920, 214, 642_000),
    ("Always-On Demand Gen", "linkedin", "fintech", "2024-Q3", 61000, 1_680_000, 8_400, 96, 288_000),
    ("Always-On Demand Gen", "linkedin", "fintech", "2024-Q4", 52000, 1_310_000, 11_130, 241, 723_000),
    ("Always-On Demand Gen", "linkedin", "insurance", "2024-Q3", 33000, 890_000, 7_120, 168, 504_000),
    ("Always-On Demand Gen", "linkedin", "banking", "2024-Q3", 41000, 1_020_000, 8_160, 191, 573_000),
    ("Brand Search Defence", "paid_search", "fintech", "2024-Q3", 27000, 410_000, 16_400, 402, 1_206_000),
    ("Brand Search Defence", "paid_search", "insurance", "2024-Q3", 19000, 288_000, 11_520, 279, 837_000),
    ("Brand Search Defence", "paid_search", "banking", "2024-Q3", 22000, 331_000, 13_240, 318, 954_000),
    ("Category Search", "paid_search", "fintech", "2024-Q3", 38000, 720_000, 14_400, 216, 648_000),
    ("Nurture Sequence", "email", "fintech", "2024-Q3", 6000, 184_000, 9_200, 276, 828_000),
    ("Nurture Sequence", "email", "insurance", "2024-Q3", 5000, 151_000, 7_550, 219, 657_000),
    ("Nurture Sequence", "email", "banking", "2024-Q3", 5500, 166_000, 8_300, 249, 747_000),
    ("Industry Roadshow", "events", "fintech", "2024-Q3", 95000, 42_000, 1_680, 84, 1_260_000),
    ("Industry Roadshow", "events", "banking", "2024-Q3", 78000, 36_000, 1_440, 71, 1_065_000),
    ("Category Search", "paid_search", "fintech", "2024-Q4", 40000, 760_000, 15_200, 243, 729_000),
    ("Nurture Sequence", "email", "fintech", "2024-Q4", 6200, 190_000, 9_500, 294, 882_000),
]

# ── the document corpus ──────────────────────────────────────────────────────
# Written with real headings, because the chunker splits on them. A brief's
# headings are its semantics: "Audience" under one campaign must never be
# merged with "Budget" of the next.

DOCUMENTS = [
    {
        "title": "Brand voice guide",
        "doc_type": "brand",
        "body": """## Who we are
We sell compliance automation software to regulated financial institutions. Our buyers are
risk and compliance leaders who are personally accountable when something goes wrong.

## Tone
Direct, calm, and specific. We write the way a trusted colleague briefs you before a board
meeting: no hedging, no hype, no exclamation marks. Short sentences. Concrete nouns.

## What we never do
We do not use fear-based messaging about regulators or fines. We do not use the words
"revolutionary", "game-changing", "seamless", or "cutting-edge". We do not use humour about
compliance failures, because our buyers have lived through them.

## Proof over adjectives
Every claim carries a number or a named customer. "Reduces review time" is weak.
"Cuts control testing time by 40% at three tier-one banks" is our register.

## Formatting
Headline under 60 characters. Opening line states the reader's problem, not our product.
One call to action per asset.""",
    },
    {
        "title": "Q3 2024 LinkedIn fintech campaign brief",
        "doc_type": "brief",
        "channel": "linkedin", "segment": "fintech", "quarter": "2024-Q3",
        "body": """## Objective
Grow qualified pipeline in the fintech segment by 25% quarter on quarter.

## Audience
Broad targeting: all financial services job titles at companies with 200+ employees in
UK and EU. We widened from the Q2 definition, which was limited to Head of Compliance and
above, in order to increase reach.

## Message
Lead with the new automated controls testing module. Product-led messaging rather than the
problem-led angle used in Q2.

## Channels and budget
LinkedIn sponsored content, 61,000 GBP. Up from 48,000 in Q2.

## Success criteria
Cost per acquisition under 300 GBP. Pipeline of at least 700,000 GBP.""",
    },
    {
        "title": "Q3 2024 LinkedIn fintech post-campaign retro",
        "doc_type": "retro",
        "channel": "linkedin", "segment": "fintech", "quarter": "2024-Q3",
        "body": """## Outcome
Missed on every success criterion. Spend rose 27% against Q2 and conversions fell 55%.

## What we think happened
Two changes landed in the same quarter and we cannot fully separate them.

First, the audience widening. We went from Head of Compliance and above to all financial
services titles at 200+ employee companies. Impressions rose 35% and click-through fell
sharply. We reached far more people who were not buyers.

Second, the message switch. Q2 led with the customer's problem and Q3 led with our product
feature. Every prior quarter where we led with the product has underperformed.

## What we would do differently
Restore the seniority filter. Return to problem-led messaging. Change one variable per
quarter so the result is attributable.

## Note for whoever reads this next
This is the third time we have widened targeting to chase reach and seen conversion fall.
It is documented in the Q1 2023 retro as well.""",
    },
    {
        "title": "Q2 2024 LinkedIn fintech campaign brief",
        "doc_type": "brief",
        "channel": "linkedin", "segment": "fintech", "quarter": "2024-Q2",
        "body": """## Objective
Establish the fintech segment as a repeatable pipeline source.

## Audience
Head of Compliance, Chief Risk Officer, and Director of Regulatory Affairs at financial
services companies with 200+ employees in UK and EU. Deliberately narrow.

## Message
Problem-led. Open on the cost of manual control testing, then introduce the product as the
response.

## Channels and budget
LinkedIn sponsored content, 48,000 GBP.

## Success criteria
CAC under 300 GBP. Achieved 224 GBP.""",
    },
    {
        "title": "Fintech ICP and persona definition",
        "doc_type": "persona",
        "segment": "fintech",
        "body": """## Ideal customer profile
Payments companies, digital banks and lending platforms with 200 to 5,000 employees,
operating under FCA or equivalent EU supervision, with an in-house compliance function of
at least four people.

## Primary persona: Head of Compliance
Accountable for control testing and regulatory reporting. Time-poor. Evaluates tools on
audit defensibility first and usability second. Sceptical of vendors who have not worked in
a regulated environment.

## What they respond to
Evidence from named peer institutions. Specific time savings. Clear answers about data
residency and audit trails.

## What loses them
Generic digital transformation language, unnamed customer references, and any suggestion
that the tool makes compliance decisions on their behalf.""",
    },
    {
        "title": "Competitor battlecard: Regulatory Systems Ltd",
        "doc_type": "battlecard",
        "body": """## Their position
Incumbent in tier-one banking. Strong audit reputation, very slow to deploy, priced for
large institutions.

## Where we win
Deployment time. We are typically live in six weeks against their nine to twelve months.
Mid-market pricing. Modern integrations.

## Where we lose
Depth of regulatory coverage outside the UK and EU. Long-standing relationships in tier-one
banks that predate our existence.

## How to talk about them
Never disparage their product. Contrast on deployment time and total cost, both of which we
can evidence. If asked directly about coverage, be honest: we cover UK and EU thoroughly and
are expanding.""",
    },
    {
        "title": "Paid search channel playbook",
        "doc_type": "playbook",
        "channel": "paid_search",
        "body": """## What this channel is for
Capturing existing demand. It does not create it. Brand defence terms convert far better
than category terms and should always be funded first.

## Structure
Separate campaigns for brand defence and category terms. Never mix them in one budget, since
brand terms will absorb it and the category performance becomes invisible.

## Benchmarks
Brand defence: CTR above 3.5%, CAC under 100 GBP.
Category: CTR above 1.8%, CAC under 250 GBP.

## Common mistake
Judging category search on the same CAC target as brand. They do different jobs.""",
    },
    {
        "title": "Email nurture playbook",
        "doc_type": "playbook",
        "channel": "email",
        "body": """## What this channel is for
Converting known contacts who are not yet ready. Cheapest channel by CAC in every quarter we
have measured, and the one most often under-resourced.

## Cadence
Five emails over three weeks. Problem, evidence, product, peer proof, offer.

## Benchmarks
Open rate above 32%, click rate above 5%, CAC under 60 GBP.

## Common mistake
Treating it as an announcement channel. Product launch blasts to the full list depress
engagement for the following two sends.""",
    },
    {
        "title": "Events channel playbook",
        "doc_type": "playbook",
        "channel": "events",
        "body": """## What this channel is for
Late-stage influence and relationship building in banking, where deals are large and slow.
Judge it on pipeline value and deal velocity, never on CAC, because the volume is small by
design and the CAC always looks terrible next to digital channels.

## Benchmarks
Pipeline per event above 500,000 GBP. Meetings booked above 40.

## Common mistake
Comparing event CAC to email CAC and concluding events do not work.""",
    },
    {
        "title": "Q1 2023 fintech retro",
        "doc_type": "retro",
        "segment": "fintech", "quarter": "2023-Q1",
        "body": """## Outcome
Underperformed on conversions despite record reach.

## Cause
We widened audience targeting mid-quarter to hit an impressions goal. Conversions fell 38%.

## Lesson
Reach is not a marketing objective. It is a diagnostic. We should stop setting impressions
targets, because they reliably produce this outcome.""",
    },
]

# ── compliance rules ─────────────────────────────────────────────────────────
# Retrieved in full rather than by similarity. There are few of them, and a
# half-retrieved rulebook is worse than none: a rule saying a claim is allowed,
# retrieved without the rule saying it needs a citation, is actively dangerous.

COMPLIANCE_RULES = [
    {"code": "C1", "severity": "high",
     "rule": "Never guarantee a regulatory outcome. Claims that our product ensures, "
             "guarantees or assures compliance, or that it will prevent enforcement action, "
             "are prohibited without exception.",
     "pattern": r"\b(guarantee|guarantees|guaranteed|ensures? compliance|assures? compliance|"
                r"prevents? (?:enforcement|fines?|penalt))"},
    {"code": "C2", "severity": "high",
     "rule": "Superlative market claims such as best, fastest, leading, number one or "
             "most accurate require a cited third-party source in the same asset.",
     "pattern": r"\b(the fastest|the best|market[- ]leading|number one|#1|the most accurate|"
                r"industry[- ]leading)\b"},
    {"code": "C3", "severity": "medium",
     "rule": "Named customers may only be referenced with a recorded approval. Use the "
             "sector and size instead, for example 'a tier-one UK bank'.",
     "pattern": r"\b(HSBC|Barclays|Lloyds|Monzo|Revolut|Starling|NatWest)\b"},
    {"code": "C4", "severity": "medium",
     "rule": "Performance figures must carry the period they were measured over. "
             "'Cuts review time by 40%' needs 'measured over a 12-month deployment'.",
     "pattern": r"\b(\d{1,3})\s?%\s?(faster|reduction|less|fewer|cut|savings?|improvement)"},
    {"code": "C5", "severity": "low",
     "rule": "Do not use fear-based framing about regulators, fines or enforcement. "
             "This is a brand rule as well as a compliance one.",
     "pattern": r"\b(avoid (?:fines|penalties)|before the regulator|regulator will|"
                r"risk (?:huge|massive) fines)"},
]

# ── the golden evaluation set ────────────────────────────────────────────────
# Small, but built the way a real one is: every case declares the route it
# should take and the documents it should retrieve, so retrieval and generation
# can be scored separately. The unanswerable slice is the important one -- a
# system that never abstains is not good, only confident.

GOLDEN_SET = [
    {
        "id": "g1", "slice": "factual", "route": "rag",
        "question": "What tone should we use in fintech campaigns?",
        "expect_documents": ["Brand voice guide"],
        "expect_contains": ["direct", "calm", "specific"],
    },
    {
        "id": "g2", "slice": "factual", "route": "rag",
        "question": "Which words are we not allowed to use in copy?",
        "expect_documents": ["Brand voice guide"],
        "expect_contains": ["revolutionary", "seamless"],
    },
    {
        "id": "g3", "slice": "factual", "route": "rag",
        "question": "How do we position against Regulatory Systems Ltd?",
        "expect_documents": ["Competitor battlecard: Regulatory Systems Ltd"],
        "expect_contains": ["deployment", "weeks"],
    },
    {
        "id": "g4", "slice": "factual", "route": "rag",
        "question": "What is our ideal customer profile in fintech?",
        "expect_documents": ["Fintech ICP and persona definition"],
        "expect_contains": ["compliance"],
    },
    {
        "id": "g5", "slice": "analytical", "route": "sql",
        "question": "What was total spend on LinkedIn in Q3 2024?",
        "expect_documents": [],
        "expect_numeric": 135000.0,
    },
    {
        "id": "g6", "slice": "analytical", "route": "sql",
        "question": "How many conversions did the fintech segment produce in Q3 2024?",
        "expect_documents": [],
        "expect_numeric": 1074.0,
    },
    {
        "id": "g7", "slice": "analytical", "route": "sql",
        "question": "Which channel had the lowest cost per acquisition in Q3 2024?",
        "expect_documents": [],
        "expect_contains": ["email"],
    },
    {
        "id": "g8", "slice": "hybrid", "route": "hybrid",
        "question": "Why did the Q3 LinkedIn fintech campaign underperform?",
        "expect_documents": ["Q3 2024 LinkedIn fintech post-campaign retro"],
        "expect_contains": ["audience", "widen"],
    },
    {
        "id": "g9", "slice": "hybrid", "route": "hybrid",
        "question": "Compare LinkedIn and email performance in Q3 2024 and explain the gap.",
        "expect_documents": ["Email nurture playbook"],
        "expect_contains": ["email"],
    },
    {
        "id": "g10", "slice": "unanswerable", "route": "rag",
        "question": "What is our TikTok strategy for the Japanese market?",
        "expect_documents": [],
        "expect_abstain": True,
    },
    {
        "id": "g11", "slice": "unanswerable", "route": "sql",
        "question": "What was our spend in Q1 2019?",
        "expect_documents": [],
        "expect_abstain": True,
    },
    {
        "id": "g12", "slice": "factual", "route": "rag",
        "question": "How should we judge the events channel?",
        "expect_documents": ["Events channel playbook"],
        "expect_contains": ["pipeline"],
    },
]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def seed_workspace() -> None:
    """Create the demo corpus if it is not already there.

    Idempotent by content hash: re-running leaves an unchanged corpus alone and
    replaces a changed document together with its chunks. That is the same
    discipline real ingestion needs, so the demo path exercises it.
    """
    conn = database.connect()
    try:
        existing = conn.execute(
            "SELECT COUNT(*) AS n FROM campaigns WHERE workspace_id = ?",
            (DEMO_WORKSPACE,)).fetchone()["n"]
        if existing == 0:
            for row in CAMPAIGNS:
                conn.execute(
                    """INSERT INTO campaigns (id, workspace_id, name, channel, segment, quarter,
                                              spend, impressions, clicks, conversions, pipeline_value)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (new_id(), DEMO_WORKSPACE, *row))

        for document in DOCUMENTS:
            digest = _hash(document["body"])
            current = conn.execute(
                "SELECT id, content_hash FROM documents WHERE workspace_id = ? AND title = ?",
                (DEMO_WORKSPACE, document["title"])).fetchone()
            if current and current["content_hash"] == digest:
                continue
            if current:
                conn.execute("DELETE FROM documents WHERE id = ?", (current["id"],))
            conn.execute(
                """INSERT INTO documents (id, workspace_id, title, doc_type, channel, segment,
                                          quarter, body, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_id(), DEMO_WORKSPACE, document["title"], document["doc_type"],
                 document.get("channel", ""), document.get("segment", ""),
                 document.get("quarter", ""), document["body"], digest))

        rules = conn.execute(
            "SELECT COUNT(*) AS n FROM compliance_rules WHERE workspace_id = ?",
            (DEMO_WORKSPACE,)).fetchone()["n"]
        if rules == 0:
            for rule in COMPLIANCE_RULES:
                conn.execute(
                    """INSERT INTO compliance_rules (id, workspace_id, code, rule, severity, pattern)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (new_id(), DEMO_WORKSPACE, rule["code"], rule["rule"],
                     rule["severity"], rule["pattern"]))
        conn.commit()
    finally:
        conn.close()
