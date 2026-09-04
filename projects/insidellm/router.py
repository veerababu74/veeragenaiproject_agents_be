"""Serving the precomputed artifacts.

Every response here is either a small static Python structure or a file that was
written at build time. Nothing is computed per request, no model is loaded, and
the example payloads are returned as the bytes already on disk rather than being
parsed into Python and re-serialised — on a two-core box that saved work is the
difference between trivial and noticeable.
"""

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response

from core.auth import current_user_id
from projects.insidellm.content import COMPARISON_ROWS, COMPONENTS, MODELS
from projects.insidellm.examples import EXAMPLES_BY_ID

logger = logging.getLogger("insidellm")

DATA_DIR = Path(__file__).resolve().parent / "data"

router = APIRouter(tags=["insidellm"])


@lru_cache(maxsize=16)
def _artifact(name: str) -> bytes:
    """Read a build artifact once and keep it. The whole dataset is well under a
    megabyte, so the cache is bounded by the number of files rather than needing
    a size budget."""
    path = DATA_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(name)
    return path.read_bytes()


def _json(payload: bytes) -> Response:
    return Response(content=payload, media_type="application/json")


@router.get("/overview")
async def overview(_: str = Depends(current_user_id)):
    """What this project contains, and the honest description of how it works."""
    return {
        "title": "Inside an LLM",
        "subtitle": "What actually happens between a sentence going in and a word coming out",
        "model": {
            "name": "GPT-2 small",
            "parameters": "124M",
            "layers": 12, "heads": 12, "d_model": 768, "vocab": 50257,
        },
        "method": (
            "Every number shown is a real forward pass over GPT-2's real pretrained weights, "
            "computed in numpy and captured layer by layer. It was run once, ahead of time, and "
            "the results were saved — so this page loads instantly and the server never holds a "
            "model in memory."
        ),
        "why_fixed_examples": (
            "The ten examples are fixed rather than typed in. Each one was chosen because it makes "
            "a specific mechanism visible, and precomputing them is what keeps the whole thing "
            "running in a few hundred megabytes."
        ),
        "components": len(COMPONENTS),
        "examples": len(EXAMPLES_BY_ID),
    }


@router.get("/components")
async def components(_: str = Depends(current_user_id)):
    """The curriculum: one entry per transformer component, in the order the data
    flows through them."""
    return {"components": sorted(COMPONENTS, key=lambda item: item["order"])}


@router.get("/models")
async def models(_: str = Depends(current_user_id)):
    return {"models": MODELS, "comparison": COMPARISON_ROWS}


@router.get("/examples")
async def list_examples(_: str = Depends(current_user_id)):
    try:
        return _json(_artifact("index"))
    except FileNotFoundError:
        raise HTTPException(503, "The precomputed data has not been built yet")


@router.get("/positional")
async def positional(_: str = Depends(current_user_id)):
    """GPT-2's learned position vectors, next to sinusoidal and rotary schemes."""
    try:
        return _json(_artifact("positional"))
    except FileNotFoundError:
        raise HTTPException(503, "The precomputed data has not been built yet")


@router.get("/examples/{example_id}")
async def get_example(example_id: str, _: str = Depends(current_user_id)):
    """One example with everything captured from its forward pass: tokenization,
    embeddings, all 144 attention heads, the worked arithmetic, the logit lens
    and the final distribution."""
    if example_id not in EXAMPLES_BY_ID:
        raise HTTPException(404, "Unknown example")
    try:
        return _json(_artifact(example_id))
    except FileNotFoundError:
        raise HTTPException(503, "This example has not been precomputed yet")
