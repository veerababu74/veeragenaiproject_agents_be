"""The evaluation harness.

**The principle the whole file is built on: score retrieval and generation
separately.** When answer quality drops you have to know which half broke. One
end-to-end number cannot tell you, and you will spend a day guessing. So every
case produces retrieval metrics computed against the documents it should have
found, and generation metrics computed against what the answer should contain,
and they are reported side by side.

**Agent metrics are a third family.** Routing accuracy, attempts used and
abstention correctness are not RAG metrics -- they measure the loop, not the
lookup -- and they are the ones that catch an agent that is technically finding
the right documents while dithering its way there.

**The abstention slice is the important one.** A system that never says "I don't
know" is not accurate, only confident. Two cases in the golden set have no
answer in the corpus, and the correct behaviour is to refuse them. They are
scored as passes when the system abstains and failures when it invents.
"""

import json
import logging
import re
import time

from projects.marketingcopilot.corpus import GOLDEN_SET
from projects.marketingcopilot.database import execute, new_id
from projects.marketingcopilot.graph import answer_question

logger = logging.getLogger("marketingcopilot.evaluation")

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[float]:
    values = []
    for raw in _NUMBER.findall(text or ""):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def _recall_at_k(retrieved_titles: list[str], expected: list[str]) -> float | None:
    """What share of the documents that should have been found, were found.

    Returns None rather than 0 when a case expects no documents. Scoring an
    abstention case as 0% recall would drag the average down for behaving
    correctly, which is how a metric ends up arguing against the thing it is
    supposed to encourage.
    """
    if not expected:
        return None
    found = sum(1 for title in expected if title in retrieved_titles)
    return found / len(expected)


def _mrr(retrieved_titles: list[str], expected: list[str]) -> float | None:
    """Reciprocal rank of the first correct document. Sensitive to ordering,
    which is what makes it the number that moves when reranking changes."""
    if not expected:
        return None
    for position, title in enumerate(retrieved_titles, start=1):
        if title in expected:
            return 1.0 / position
    return 0.0


def _contains_all(answer: str, needles: list[str]) -> bool:
    lowered = (answer or "").lower()
    return all(needle.lower() in lowered for needle in needles)


def _looks_like_abstention(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(phrase in lowered for phrase in (
        "don't have", "do not have", "no information", "not in the current workspace",
        "found nothing", "cannot answer", "couldn't find", "could not find"))


def evaluate_case(case: dict, config: dict) -> dict:
    """Run one golden case and score it on all three families."""
    started = time.time()
    try:
        result = answer_question(case["question"], config)
    except Exception as error:  # noqa: BLE001 — a failed case is a result, not a crash
        return {
            "id": case["id"], "slice": case["slice"], "question": case["question"],
            "error": str(error)[:300], "passed": False,
            "latency_ms": int((time.time() - started) * 1000),
        }

    titles = [citation["title"] for citation in result["citations"]]
    expected = case.get("expect_documents", [])

    recall = _recall_at_k(titles, expected)
    mrr = _mrr(titles, expected)
    routed_correctly = result["route"] == case["route"]

    # ── did the answer do its job? ───────────────────────────────────────────
    if case.get("expect_abstain"):
        answered_well = result["abstained"] or _looks_like_abstention(result["answer"])
        reason = "abstained" if answered_well else "answered a question it could not answer"
    elif "expect_numeric" in case:
        # Numeric cases are checked by looking for the value anywhere in the
        # answer, with a tolerance. Exact string matching fails on formatting
        # ("135,000" vs "135000" vs "£135k") without meaning anything.
        target = case["expect_numeric"]
        answered_well = any(abs(value - target) <= max(1.0, abs(target) * 0.01)
                            for value in _numbers(result["answer"]))
        reason = f"expected ≈{target:,.0f}"
    else:
        answered_well = _contains_all(result["answer"], case.get("expect_contains", []))
        reason = f"expected to mention {case.get('expect_contains')}"

    grounded = bool(result["citations"]) or result["route"] in ("sql", "generate")
    if case.get("expect_abstain"):
        grounded = True  # abstaining without citations is correct, not ungrounded

    return {
        "id": case["id"],
        "slice": case["slice"],
        "question": case["question"],
        "expected_route": case["route"],
        "actual_route": result["route"],
        "routed_correctly": routed_correctly,
        "recall_at_k": recall,
        "mrr": mrr,
        "retrieved": titles,
        "expected_documents": expected,
        "answered_well": answered_well,
        "grounded": grounded,
        "abstained": result["abstained"],
        "attempts": result["attempts"],
        "sql_query": result["sql_query"],
        "answer": result["answer"][:600],
        "reason": reason,
        # A case passes only if it both took a sensible path and produced the
        # right answer. Either alone is not success.
        "passed": bool(answered_well and grounded),
        "latency_ms": result["latency_ms"],
    }


def _mean(values: list) -> float | None:
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 4) if present else None


def run_evaluation(config: dict, user_id: str, limit: int | None = None) -> dict:
    """Run the golden set and persist the result against the run.

    Stored rather than printed, because the value of an eval is the comparison
    between runs. A single score tells you nothing; a score next to last week's
    tells you whether the change you shipped helped.
    """
    cases = GOLDEN_SET[:limit] if limit else GOLDEN_SET
    results = [evaluate_case(case, config) for case in cases]

    by_slice: dict[str, list[dict]] = {}
    for result in results:
        by_slice.setdefault(result["slice"], []).append(result)

    metrics = {
        # Retrieval
        "recall_at_k": _mean([r.get("recall_at_k") for r in results]),
        "mrr": _mean([r.get("mrr") for r in results]),
        # Generation
        "answer_quality": _mean([1.0 if r.get("answered_well") else 0.0 for r in results]),
        "grounded_rate": _mean([1.0 if r.get("grounded") else 0.0 for r in results]),
        # Agent
        "routing_accuracy": _mean([1.0 if r.get("routed_correctly") else 0.0 for r in results]),
        "mean_attempts": _mean([r.get("attempts", 0) for r in results]),
        "abstention_correct": _mean([
            1.0 if r.get("answered_well") else 0.0
            for r in results if r["slice"] == "unanswerable"]),
        # End to end
        "pass_rate": _mean([1.0 if r.get("passed") else 0.0 for r in results]),
        "p50_latency_ms": sorted(r["latency_ms"] for r in results)[len(results) // 2],
        "cases": len(results),
    }
    slices = {
        name: {
            "cases": len(rows),
            "pass_rate": _mean([1.0 if r.get("passed") else 0.0 for r in rows]),
            "recall_at_k": _mean([r.get("recall_at_k") for r in rows]),
            "routing_accuracy": _mean([1.0 if r.get("routed_correctly") else 0.0 for r in rows]),
        }
        for name, rows in by_slice.items()
    }

    run_id = new_id()
    execute(
        "INSERT INTO eval_runs (id, user_id, dataset, metrics, cases) VALUES (?, ?, ?, ?, ?)",
        (run_id, user_id, "golden-v1",
         json.dumps({**metrics, "slices": slices}), json.dumps(results)))

    return {"run_id": run_id, "metrics": metrics, "slices": slices, "results": results}


# What each number means, served to the UI beside the number itself. A metric
# nobody can interpret is a metric nobody acts on.
METRIC_GLOSSARY = [
    {"id": "recall_at_k", "family": "retrieval", "name": "Recall@k",
     "formula": "found ∩ expected / expected",
     "means": "Did the right document reach the top k at all? This is the ceiling on "
              "everything downstream — if it was not retrieved, no prompt can fix it. "
              "Watch this one first."},
    {"id": "mrr", "family": "retrieval", "name": "MRR",
     "formula": "mean of 1 / rank(first correct)",
     "means": "How high up the first correct document landed. Sensitive to ordering, so "
              "this is the number that moves when reranking changes."},
    {"id": "answer_quality", "family": "generation", "name": "Answer quality",
     "formula": "share of cases whose answer met its expectation",
     "means": "Contains the expected facts, the expected number within tolerance, or "
              "correctly abstains — whichever the case asked for."},
    {"id": "grounded_rate", "family": "generation", "name": "Grounded rate",
     "formula": "share of answers backed by citations or a tool result",
     "means": "The anti-hallucination check. An answer with no citation and no query "
              "behind it came from somewhere the corpus cannot vouch for."},
    {"id": "routing_accuracy", "family": "agent", "name": "Routing accuracy",
     "formula": "share of cases routed as expected",
     "means": "Did the agent pick the right path? A misroute makes every downstream "
              "metric meaningless, so this is read before the others."},
    {"id": "mean_attempts", "family": "agent", "name": "Mean attempts",
     "formula": "average retrieval rewrites used",
     "means": "How much the self-correction loop is working. Rising values are an early "
              "warning that retrieval is decaying, before anyone complains."},
    {"id": "abstention_correct", "family": "agent", "name": "Abstention",
     "formula": "share of unanswerable cases correctly refused",
     "means": "A system that never says 'I don't know' is not accurate, only confident. "
              "These cases have no answer in the corpus and refusing them is the pass."},
    {"id": "pass_rate", "family": "end-to-end", "name": "Pass rate",
     "formula": "answered well AND grounded",
     "means": "Both halves right. Either alone is not success."},
]
