"""Every endpoint the Marketing Copilot exposes.

The user brings their own key. Nothing here holds a platform model credential:
setup stores the key against the user id, and every call that needs a model
reads it back. That keeps the project free to run and makes the cost the user's
own, which is the same arrangement the other labs use.
"""

import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth import current_user_id
from projects.marketingcopilot import content, evaluation, examples as example_catalog, providers, retrieval
from projects.marketingcopilot.corpus import GOLDEN_SET
from projects.marketingcopilot.database import (
    DEMO_WORKSPACE, execute, get_settings_row, new_id, query,
    record_message, save_settings_row,
)
from projects.marketingcopilot.graph import answer_question
from projects.marketingcopilot.tools import check_compliance, load_rules

logger = logging.getLogger("marketingcopilot")

router = APIRouter(tags=["marketingcopilot"])


# ── models ───────────────────────────────────────────────────────────────────

class SetupRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=30)
    chat_model: str = Field(min_length=1, max_length=80)
    embed_model: str = Field(min_length=1, max_length=80)
    api_key: str = Field(min_length=8, max_length=400)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str = Field(default="", max_length=64)


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1, max_length=64)
    rating: int = Field(ge=-1, le=1)
    reason: str = Field(default="", max_length=500)


class ComplianceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class EvalRequest(BaseModel):
    limit: int = Field(default=0, ge=0, le=50)


def _config(user_id: str) -> dict:
    row = get_settings_row(user_id)
    if not row or not row["api_key"]:
        raise HTTPException(400, "Add your API key in Setup before running the copilot.")
    return {
        "provider": row["provider"], "chat_model": row["chat_model"],
        "embed_model": row["embed_model"], "api_key": row["api_key"],
    }


# ── the explanation ──────────────────────────────────────────────────────────

@router.get("/overview")
async def overview(_: str = Depends(current_user_id)):
    documents = query(
        "SELECT COUNT(*) AS n FROM documents WHERE workspace_id = ?", (DEMO_WORKSPACE,))[0]["n"]
    campaigns = query(
        "SELECT COUNT(*) AS n FROM campaigns WHERE workspace_id = ?", (DEMO_WORKSPACE,))[0]["n"]
    indexed = query(
        "SELECT COUNT(*) AS n FROM chunks WHERE workspace_id = ?", (DEMO_WORKSPACE,))[0]["n"]
    return {
        **content.OVERVIEW,
        "corpus": {"documents": documents, "campaigns": campaigns, "chunks_indexed": indexed,
                   "rules": len(load_rules()),
                   "retriever": "pinecone" if retrieval.pinecone_index() else "local"},
    }


@router.get("/explain")
async def explain(_: str = Depends(current_user_id)):
    """The graph, node by node, and the failures that shaped it."""
    return {
        "nodes": sorted(content.NODES, key=lambda item: item["order"]),
        "failure_modes": content.FAILURE_MODES,
        "metrics": evaluation.METRIC_GLOSSARY,
    }


# ── setup ────────────────────────────────────────────────────────────────────

@router.get("/setup")
async def read_setup(user_id: str = Depends(current_user_id)):
    row = get_settings_row(user_id)
    return {
        "providers": providers.catalog(),
        "configured": bool(row and row["api_key"]),
        "provider": row["provider"] if row else "openai",
        "chat_model": row["chat_model"] if row else "gpt-4o-mini",
        "embed_model": row["embed_model"] if row else "text-embedding-3-small",
        # Never return the key. A masked hint is enough to confirm which one is saved.
        "key_hint": f"…{row['api_key'][-4:]}" if row and row["api_key"] else "",
    }


@router.post("/setup")
async def write_setup(request: SetupRequest, user_id: str = Depends(current_user_id)):
    if not providers.known(request.provider):
        raise HTTPException(400, f"Unknown provider: {request.provider}")
    save_settings_row(user_id, request.provider, request.chat_model,
                      request.embed_model, request.api_key)
    return {"saved": True}


@router.post("/index")
async def build_index(user_id: str = Depends(current_user_id)):
    """Chunk and embed the corpus with the user's key.

    Synchronous because the demo corpus is ten documents and finishes in a few
    seconds. A real corpus would return a job id and index in the background —
    an HTTP request should not hold an embedding run.
    """
    config = _config(user_id)
    try:
        result = retrieval.index_workspace(
            config["provider"], config["embed_model"], config["api_key"])
    except Exception as error:  # noqa: BLE001 — surface the provider's own message
        raise HTTPException(400, f"Indexing failed: {error}") from error
    return result


# ── the corpus, so the user can see what is being searched ───────────────────

@router.get("/corpus")
async def corpus(_: str = Depends(current_user_id)):
    documents = query(
        """SELECT id, title, doc_type, channel, segment, quarter, chunk_count,
                  length(body) AS characters
           FROM documents WHERE workspace_id = ? ORDER BY doc_type, title""",
        (DEMO_WORKSPACE,))
    campaigns = query(
        """SELECT name, channel, segment, quarter, spend, impressions, clicks,
                  conversions, pipeline_value
           FROM campaigns WHERE workspace_id = ?
           ORDER BY quarter DESC, channel, segment""", (DEMO_WORKSPACE,))
    return {"documents": documents, "campaigns": campaigns, "rules": load_rules()}


@router.get("/corpus/{document_id}")
async def document(document_id: str, _: str = Depends(current_user_id)):
    rows = query("SELECT * FROM documents WHERE id = ? AND workspace_id = ?",
                 (document_id, DEMO_WORKSPACE))
    if not rows:
        raise HTTPException(404, "Unknown document")
    chunks = query(
        "SELECT id, ordinal, section, text FROM chunks WHERE document_id = ? ORDER BY ordinal",
        (document_id,))
    return {**rows[0], "chunks": chunks}


# ── chat ─────────────────────────────────────────────────────────────────────

@router.get("/examples")
async def examples(_: str = Depends(current_user_id)):
    """Ready-made scenarios, one per route.

    Offered instead of a blank chat box because the interesting thing about this
    system is that different questions take different paths, and that is
    invisible until you have run one of each. Each scenario says what it
    demonstrates and what to watch for while it runs.
    """
    return {"examples": sorted(example_catalog.EXAMPLES, key=lambda item: item["order"])}


@router.get("/examples/{example_id}")
async def example_detail(example_id: str, _: str = Depends(current_user_id)):
    found = example_catalog.example_by_id(example_id)
    if not found:
        raise HTTPException(404, "Unknown example")
    return found


@router.get("/suggestions")
async def suggestions(_: str = Depends(current_user_id)):
    """The short form the composer uses. Kept as a thin view over the scenarios
    so the two can never disagree about what the copilot is good at."""
    return {"suggestions": [
        {"text": item["question"], "route": item["route"], "note": item["tagline"]}
        for item in sorted(example_catalog.EXAMPLES, key=lambda entry: entry["order"])
    ]}


@router.post("/chat")
async def chat(request: ChatRequest, user_id: str = Depends(current_user_id)):
    """Run the graph and record everything worth debugging later."""
    config = _config(user_id)

    conversation_id = request.conversation_id
    if conversation_id:
        owned = query("SELECT id FROM conversations WHERE id = ? AND user_id = ?",
                      (conversation_id, user_id))
        if not owned:
            raise HTTPException(404, "Unknown conversation")
    else:
        conversation_id = new_id()
        execute("INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
                (conversation_id, user_id, request.message[:80]))

    history = query(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at LIMIT 10",
        (conversation_id,))

    record_message(conversation_id=conversation_id, user_id=user_id,
                   role="user", content=request.message)

    started = time.time()
    try:
        result = answer_question(request.message, config, history)
    except Exception as error:  # noqa: BLE001 — report, do not 500 on a provider hiccup
        logger.exception("graph failed")
        raise HTTPException(502, f"The agent failed: {error}") from error

    message_id = record_message(
        conversation_id=conversation_id, user_id=user_id, role="assistant",
        content=result["answer"], route=result["route"], citations=result["citations"],
        steps=result["steps"], sql_query=result["sql_query"], attempts=result["attempts"],
        latency_ms=result["latency_ms"], top_score=result["top_score"])

    return {
        "conversation_id": conversation_id,
        "message_id": message_id,
        **result,
        "elapsed_ms": int((time.time() - started) * 1000),
    }


@router.get("/conversations")
async def conversations(user_id: str = Depends(current_user_id)):
    return {"conversations": query(
        """SELECT c.id, c.title, c.created_at, COUNT(m.id) AS messages
           FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
           WHERE c.user_id = ? GROUP BY c.id ORDER BY c.created_at DESC LIMIT 30""",
        (user_id,))}


@router.get("/conversations/{conversation_id}")
async def conversation(conversation_id: str, user_id: str = Depends(current_user_id)):
    rows = query(
        """SELECT * FROM messages WHERE conversation_id = ? AND user_id = ?
           ORDER BY created_at""", (conversation_id, user_id))
    if not rows:
        raise HTTPException(404, "Unknown conversation")
    for row in rows:
        row["citations"] = json.loads(row["citations"] or "[]")
        row["steps"] = json.loads(row["steps"] or "[]")
    return {"messages": rows}


@router.post("/feedback")
async def feedback(request: FeedbackRequest, user_id: str = Depends(current_user_id)):
    owned = query("SELECT id FROM messages WHERE id = ? AND user_id = ?",
                  (request.message_id, user_id))
    if not owned:
        raise HTTPException(404, "Unknown message")
    execute("INSERT INTO feedback (id, message_id, user_id, rating, reason) VALUES (?, ?, ?, ?, ?)",
            (new_id(), request.message_id, user_id, request.rating, request.reason))
    return {"recorded": True}


# ── compliance, on demand ────────────────────────────────────────────────────

@router.post("/compliance/check")
async def compliance_check(request: ComplianceRequest, _: str = Depends(current_user_id)):
    """Paste any copy and see which rules it trips.

    Exposed on its own because it is useful without the rest of the system, and
    because seeing the rulebook applied to your own text is the fastest way to
    understand what the guardrail does and does not cover.
    """
    return check_compliance(request.text)


# ── evaluation ───────────────────────────────────────────────────────────────

@router.get("/eval/dataset")
async def eval_dataset(_: str = Depends(current_user_id)):
    return {
        "cases": [{k: v for k, v in case.items() if k != "expect_contains"} |
                  {"expect_contains": case.get("expect_contains", [])}
                  for case in GOLDEN_SET],
        "metrics": evaluation.METRIC_GLOSSARY,
        "note": ("Every case declares the route it should take and the documents it should "
                 "find, so retrieval and generation are scored separately. When quality drops "
                 "you need to know which half broke."),
    }


@router.post("/eval/run")
async def eval_run(request: EvalRequest, user_id: str = Depends(current_user_id)):
    """Run the golden set. Costs real tokens, so the size is capped."""
    config = _config(user_id)
    return evaluation.run_evaluation(config, user_id, request.limit or None)


@router.get("/eval/runs")
async def eval_runs(user_id: str = Depends(current_user_id)):
    rows = query(
        "SELECT id, dataset, metrics, created_at FROM eval_runs WHERE user_id = ?"
        " ORDER BY created_at DESC LIMIT 20", (user_id,))
    for row in rows:
        row["metrics"] = json.loads(row["metrics"] or "{}")
    return {"runs": rows}


@router.get("/eval/runs/{run_id}")
async def eval_run_detail(run_id: str, user_id: str = Depends(current_user_id)):
    rows = query("SELECT * FROM eval_runs WHERE id = ? AND user_id = ?", (run_id, user_id))
    if not rows:
        raise HTTPException(404, "Unknown run")
    run = rows[0]
    run["metrics"] = json.loads(run["metrics"] or "{}")
    run["cases"] = json.loads(run["cases"] or "[]")
    return run


# ── monitoring ───────────────────────────────────────────────────────────────

@router.get("/monitor")
async def monitor(user_id: str = Depends(current_user_id)):
    """What you would put on a dashboard, computed from the message table.

    The three that matter are here and are worth naming: the distribution of top
    retrieval scores, how often the loop exhausts its attempts, and the
    thumbs-down rate joined back to what those answers cited. Bad RAG does not
    throw exceptions — it returns confident nonsense with a 200 — so the first
    of those is the one that catches quality decay before a user complains.
    """
    totals = query(
        """SELECT COUNT(*) AS answers,
                  COALESCE(AVG(latency_ms), 0) AS avg_latency,
                  COALESCE(AVG(top_score), 0) AS avg_top_score,
                  COALESCE(AVG(attempts), 0) AS avg_attempts,
                  SUM(CASE WHEN attempts >= 2 THEN 1 ELSE 0 END) AS loop_exhausted
           FROM messages WHERE user_id = ? AND role = 'assistant'""", (user_id,))[0]

    by_route = query(
        """SELECT route, COUNT(*) AS n, ROUND(AVG(latency_ms)) AS avg_latency,
                  ROUND(AVG(top_score), 4) AS avg_top_score
           FROM messages WHERE user_id = ? AND role = 'assistant' AND route != ''
           GROUP BY route ORDER BY n DESC""", (user_id,))

    latencies = [row["latency_ms"] for row in query(
        "SELECT latency_ms FROM messages WHERE user_id = ? AND role = 'assistant'"
        " ORDER BY latency_ms", (user_id,))]
    percentile = (lambda p: latencies[min(len(latencies) - 1, int(len(latencies) * p))]
                  if latencies else 0)

    ratings = query(
        """SELECT rating, COUNT(*) AS n FROM feedback WHERE user_id = ?
           GROUP BY rating""", (user_id,))

    # The query the section exists for: every negative answer with what it cited.
    negatives = query(
        """SELECT m.id, m.content, m.route, m.top_score, m.attempts, m.citations, f.reason
           FROM feedback f JOIN messages m ON m.id = f.message_id
           WHERE f.user_id = ? AND f.rating < 0
           ORDER BY f.created_at DESC LIMIT 10""", (user_id,))
    for row in negatives:
        row["citations"] = json.loads(row["citations"] or "[]")
        row["content"] = row["content"][:400]

    return {
        "totals": {
            "answers": totals["answers"],
            "avg_latency_ms": round(totals["avg_latency"]),
            "p50_latency_ms": percentile(0.5),
            "p95_latency_ms": percentile(0.95),
            "avg_top_score": round(totals["avg_top_score"], 4),
            "avg_attempts": round(totals["avg_attempts"], 2),
            "loop_exhausted": totals["loop_exhausted"] or 0,
        },
        "by_route": by_route,
        "feedback": {("positive" if row["rating"] > 0 else "negative"): row["n"]
                     for row in ratings},
        "negatives": negatives,
        "alerts": [
            {"metric": "avg_top_score", "watch": "a falling median means retrieval decay",
             "why": "It fires before users complain, because bad retrieval returns a 200."},
            {"metric": "loop_exhausted", "watch": "a rising count means the agent cannot find context",
             "why": "It is failing and spending money to do it."},
            {"metric": "p95_latency_ms", "watch": "the tail, not the median",
             "why": "p50 is not the experience; the slow requests are."},
        ],
    }
