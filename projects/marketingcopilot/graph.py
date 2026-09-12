"""The agent, as an explicit LangGraph state machine.

The shape is a router with a self-correcting retrieval loop:

    route ──► retrieve ──► grade ──► (rewrite ─┐)
          ├─► sql            │       ▲─────────┘  bounded at MAX_ATTEMPTS
          ├─► hybrid ────────┘
          └─► generate
                  └──► synthesise ──► compliance ──► respond

**Why a graph rather than a free ReAct loop.** Every node is a plain function I
can call in a test, the control flow is data rather than prose in a prompt, and
the path a request took is recoverable afterwards because each node appends to
`steps`. A free-form loop is harder to bound and much harder to explain when it
misbehaves.

**Why most traffic never enters the loop.** An agent is a loop with a decision in
it, and every loop is latency, cost and a new way to fail. So the *routing* is
the intelligent part and the common paths stay dumb: a factual question is one
retrieval and one generation, and a numeric question is one tool call. The loop
exists for the minority of questions that genuinely need it.

**Why the loop is bounded.** An unbounded self-correction loop is a bill and an
outage. `attempts` is capped, and the behaviour at the cap is defined: say what
was not found rather than keep spending. "I don't know" is a feature that has to
be built.
"""

import json
import logging
import re
import time
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from projects.marketingcopilot import retrieval
from projects.marketingcopilot.providers import build_llm
from projects.marketingcopilot.tools import (
    SQLError, check_compliance, generate_sql, run_sql,
)

logger = logging.getLogger("marketingcopilot.graph")

# Two retrieval attempts, then stop. Chosen rather than tuned: a third rewrite
# almost never rescues a question whose answer is simply not in the corpus, and
# the cost of finding that out is paid on every unanswerable question.
MAX_ATTEMPTS = 2

Route = Literal["rag", "sql", "hybrid", "generate"]


def _merge_steps(existing: list, incoming: list) -> list:
    return (existing or []) + (incoming or [])


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    route: Route
    filters: dict
    context: list[dict]
    sql_query: str
    sql_rows: list[dict]
    sql_error: str
    answer: str
    citations: list[dict]
    compliance: dict
    attempts: int
    abstained: bool
    # Appended by every node, so the UI can show the path as it happens and the
    # database can store what actually ran.
    steps: Annotated[list[dict], _merge_steps]
    config: dict


def _step(name: str, detail: str, **extra) -> dict:
    return {"node": name, "detail": detail, "at": round(time.time(), 3), **extra}


def _llm(state: AgentState, temperature: float = 0.2, max_tokens: int = 1600):
    config = state["config"]
    return build_llm(config["provider"], config["chat_model"], config["api_key"],
                     temperature=temperature, max_tokens=max_tokens)


def _history_text(state: AgentState, limit: int = 4) -> str:
    turns = (state.get("history") or [])[-limit:]
    return "\n".join(f"{turn['role']}: {turn['content'][:400]}" for turn in turns)


# ── nodes ────────────────────────────────────────────────────────────────────

def route_question(state: AgentState) -> dict:
    """Classify into one of four paths.

    A small, cold model, and a constrained output. On anything ambiguous it
    falls back to `hybrid`, because doing both is slower and being wrong is
    worse -- a misroute makes every downstream metric meaningless.
    """
    question = state["question"]
    history = _history_text(state)
    prompt = f"""Classify a marketing team's question into exactly one route.

rag       — asks about documented knowledge: brand voice, briefs, personas,
            competitors, playbooks, past retros. No arithmetic.
sql       — asks for numbers: spend, conversions, CTR, CAC, ROAS, comparisons
            between channels/segments/quarters, "how much", "which performed best".
hybrid    — needs numbers AND documented context: "why did X underperform",
            "compare A and B and explain", anything asking for a cause.
generate  — asks to write or draft something: copy, an email, a brief, a headline.

{f"Recent conversation:{chr(10)}{history}{chr(10)}" if history else ""}Question: {question}

Answer with one word only: rag, sql, hybrid, or generate."""

    raw = _llm(state, temperature=0.0, max_tokens=10).invoke(prompt).content.strip().lower()
    route: Route = next((r for r in ("hybrid", "generate", "sql", "rag") if r in raw), "hybrid")
    filters = retrieval.infer_filters(question)
    return {
        "route": route,
        "filters": filters,
        "attempts": 0,
        "steps": [_step("route", f"classified as {route}", route=route, filters=filters)],
    }


def rewrite_question(state: AgentState) -> dict:
    """Resolve the question against history into something standalone.

    Embedding a pronoun retrieves noise: "draft a revised one" has no content to
    match on. This runs before the first retrieval on multi-turn conversations
    and again when the grader rejects what came back.
    """
    attempts = state.get("attempts", 0)
    history = _history_text(state)
    reason = ("The previous search returned nothing relevant. Rewrite it to use different "
              "wording and broader terms." if attempts else
              "Rewrite it as a standalone question that needs no conversation history.")
    prompt = (f"{reason}\n\n"
              f"{f'Conversation:{chr(10)}{history}{chr(10)}' if history else ''}"
              f"Question: {state['question']}\n\nRewritten question:")
    rewritten = _llm(state, temperature=0.0, max_tokens=120).invoke(prompt).content.strip()
    return {
        "question": rewritten or state["question"],
        "attempts": attempts + 1,
        "steps": [_step("rewrite", f"attempt {attempts + 1}: {rewritten}")],
    }


def retrieve(state: AgentState) -> dict:
    config = state["config"]
    # The embedding half, not the chat half: retrieval must use the same model
    # the corpus was indexed with, whatever the chat model happens to be.
    hits = retrieval.search(
        state["question"], config["embed_provider"], config["embed_model"],
        config["embed_api_key"], top_k=5, filters=state.get("filters") or {})
    detail = (f"{len(hits)} chunks from {len({h.get('title') for h in hits})} documents"
              if hits else "nothing matched")
    return {
        "context": hits,
        "steps": [_step("retrieve", detail,
                        filters=state.get("filters") or {},
                        titles=[h.get("title") for h in hits],
                        top_score=round(hits[0].get("fused_score", 0), 5) if hits else 0)],
    }


def grade_context(state: AgentState) -> dict:
    """Ask whether the retrieved context can actually answer the question.

    This is the self-correcting step (the CRAG pattern). Grading the *context*
    rather than the answer is the point: it catches a retrieval failure before a
    fluent answer is written from irrelevant chunks, which is the failure mode
    that produces confident nonsense with a 200.
    """
    context = state.get("context") or []
    if not context:
        return {"steps": [_step("grade", "no context to grade", relevant=False)]}

    excerpts = "\n\n".join(
        f"[{index + 1}] {hit.get('title', '')} — {hit['text'][:500]}"
        for index, hit in enumerate(context))
    prompt = (f"Question: {state['question']}\n\nRetrieved passages:\n{excerpts}\n\n"
              "Can the question be answered using only these passages? "
              "Answer YES or NO, then a five-word reason.")
    verdict = _llm(state, temperature=0.0, max_tokens=40).invoke(prompt).content.strip()
    relevant = verdict.lower().startswith("yes")
    return {"steps": [_step("grade", verdict[:120], relevant=relevant)]}


def run_sql_node(state: AgentState) -> dict:
    """Generate SQL, validate it, run it read-only.

    A failure here is reported rather than swallowed. The generated SQL travels
    with the answer either way, because a number nobody can check is a number
    nobody should use.
    """
    llm = _llm(state, temperature=0.0, max_tokens=400)
    try:
        statement = generate_sql(llm, state["question"])
        rows = run_sql(statement)
        detail = f"{len(rows)} row(s)" if rows else "no rows matched"
        return {
            "sql_query": statement,
            "sql_rows": rows,
            "steps": [_step("sql", detail, sql=statement, rows=len(rows))],
        }
    except SQLError as error:
        return {
            "sql_error": str(error),
            "sql_rows": [],
            "steps": [_step("sql", f"rejected: {error}", error=str(error))],
        }


def synthesise(state: AgentState) -> dict:
    """Write the answer from whatever the earlier nodes gathered.

    The instruction that matters is the one telling it to abstain. Everything
    else about grounding follows from having the context; abstention has to be
    asked for explicitly or the model will always find something to say.
    """
    context = state.get("context") or []
    rows = state.get("sql_rows") or []
    route = state.get("route", "rag")
    exhausted = state.get("attempts", 0) >= MAX_ATTEMPTS and not context and not rows

    if exhausted or (not context and not rows and route != "generate"):
        answer = (
            "I don't have information about that in the current workspace. "
            + (f"I looked for documents matching {state.get('filters')} and found none. "
               if state.get("filters") else "I searched the document corpus and the campaign "
               "database and found nothing relevant. ")
            + "The corpus covers brand voice, LinkedIn/paid search/email/events playbooks, "
              "fintech and banking campaign briefs and retros for 2024.")
        return {
            "answer": answer, "citations": [], "abstained": True,
            "steps": [_step("synthesise", "abstained — nothing relevant found", abstained=True)],
        }

    blocks = []
    if context:
        blocks.append("Documents:\n" + "\n\n".join(
            f"[{index + 1}] {hit.get('title', '')} ({hit.get('doc_type', '')})\n{hit['text'][:900]}"
            for index, hit in enumerate(context)))
    if rows:
        blocks.append("Campaign data (from SQL):\n" + json.dumps(rows[:25], indent=1))
    if state.get("sql_error"):
        blocks.append(f"Note: the data query failed — {state['sql_error']}")

    instruction = {
        "generate": "Write the requested asset. Follow the brand voice document exactly. "
                    "Every claim must carry a number or a named source.",
        "sql": "Answer with the numbers. State the figures plainly and say what they are "
               "derived from.",
        "hybrid": "Explain what happened. Use the numbers for what, and the documents for why. "
                  "Be explicit about which parts are evidenced and which are inference.",
        "rag": "Answer from the documents.",
    }[route]

    prompt = f"""You are a marketing copilot for a B2B SaaS company selling to financial services.

{instruction}

Rules:
- Use only the material below. Never invent a number, a customer name or a date.
- Cite documents inline as [1], [2] matching the numbering.
- If the material does not answer part of the question, say which part is missing.
- Be concise and specific. No hedging, no filler.

{chr(10).join(blocks)}

Question: {state['question']}

Answer:"""

    answer = _llm(state, temperature=0.3 if route == "generate" else 0.1).invoke(prompt).content
    citations = [{
        "n": index + 1,
        "chunk_id": hit.get("id"),
        "document_id": hit.get("document_id"),
        "title": hit.get("title"),
        "doc_type": hit.get("doc_type"),
        "section": hit.get("section"),
        "score": hit.get("fused_score", 0),
    } for index, hit in enumerate(context)]

    return {
        "answer": answer,
        "citations": citations,
        "abstained": False,
        "steps": [_step("synthesise", f"{len(answer)} chars, {len(citations)} citation(s)")],
    }


def compliance_node(state: AgentState) -> dict:
    """Check the draft against every rule before it is returned.

    Only generated assets are checked. Running it over a factual answer produces
    noise -- quoting a retro that contains the word "guarantee" is not a
    marketing claim -- and a guardrail that cries wolf gets ignored.
    """
    if state.get("route") != "generate" or state.get("abstained"):
        return {"steps": [_step("compliance", "skipped — not a generated asset")]}

    result = check_compliance(state.get("answer", ""))
    codes = ", ".join(v["code"] for v in result["violations"]) or "none"
    return {
        "compliance": result,
        "steps": [_step("compliance",
                        "passed" if result["passed"] else f"violations: {codes}",
                        passed=result["passed"], violations=result["violations"])],
    }


# ── edges ────────────────────────────────────────────────────────────────────

def after_route(state: AgentState) -> str:
    return {"sql": "sql", "generate": "retrieve", "rag": "retrieve", "hybrid": "retrieve"}[
        state.get("route", "rag")]


def after_grade(state: AgentState) -> str:
    """Retry, move on to SQL, or synthesise.

    The bound lives here. Once `attempts` reaches MAX_ATTEMPTS the graph stops
    trying to retrieve and goes to synthesise, which knows how to abstain.
    """
    last = (state.get("steps") or [])[-1]
    relevant = last.get("relevant", False)
    route = state.get("route", "rag")

    if not relevant and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "rewrite"
    if route == "hybrid":
        return "sql"
    return "synthesise"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", route_question)
    graph.add_node("rewrite", rewrite_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade_context)
    graph.add_node("sql", run_sql_node)
    graph.add_node("synthesise", synthesise)
    graph.add_node("compliance", compliance_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, {"retrieve": "retrieve", "sql": "sql"})
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, {
        "rewrite": "rewrite", "sql": "sql", "synthesise": "synthesise"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("sql", "synthesise")
    graph.add_edge("synthesise", "compliance")
    graph.add_edge("compliance", END)
    return graph.compile()


_COMPILED = None


def compiled():
    """One compiled graph for the process. Compilation is pure structure -- the
    user's key and models travel in state, so a single instance serves everyone."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def answer_question(question: str, config: dict, history: list[dict] | None = None) -> dict:
    """Run the graph once and return everything worth recording."""
    started = time.time()
    state = compiled().invoke({
        "question": question,
        "history": history or [],
        "config": config,
        "steps": [],
        "attempts": 0,
    })
    context = state.get("context") or []
    return {
        "answer": state.get("answer", ""),
        "route": state.get("route", ""),
        "citations": state.get("citations", []),
        "steps": state.get("steps", []),
        "sql_query": state.get("sql_query", ""),
        "sql_rows": state.get("sql_rows", []),
        "compliance": state.get("compliance"),
        "attempts": state.get("attempts", 0),
        "abstained": state.get("abstained", False),
        "resolved_question": state.get("question", question),
        "top_score": round(context[0].get("fused_score", 0), 5) if context else 0.0,
        "latency_ms": int((time.time() - started) * 1000),
    }
