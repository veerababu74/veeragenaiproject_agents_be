"""Running the agent and reading back what happened."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from core.auth import current_user_id
from projects.simpleagent.database import get_db
from projects.simpleagent.models import RunRequest
from projects.simpleagent.services.runner import run_summary, stream_run
from projects.simpleagent.services.tracing import load_steps

router = APIRouter(tags=["runs"])


@router.post("/run/stream")
async def run_stream(request: RunRequest, user_id: str = Depends(current_user_id)):
    """Server-sent events for one run: a `start`, a `step` for every decision,
    tool call and tool result, a `token` per chunk of the reply, then `done`."""

    async def events():
        try:
            async for event in stream_run(user_id, request.message, request.conversation_id):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as error:
            yield f'data: {json.dumps({"type": "error", "error": str(error)})}\n\n'

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # stops proxies buffering the stream into one blob
    })


@router.get("/runs")
async def list_runs(limit: int = 20, user_id: str = Depends(current_user_id)):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM runs WHERE user_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (user_id, max(1, min(limit, 50)))).fetchall()
    conn.close()
    return {"runs": [dict(row) for row in rows]}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, user_id: str = Depends(current_user_id)):
    conn = get_db()
    row = conn.execute("SELECT * FROM runs WHERE id=? AND user_id=?", (run_id, user_id)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Run not found")
    steps = load_steps(user_id, run_id)
    return {"run": dict(row), "steps": steps, "summary": run_summary(steps)}


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user_id: str = Depends(current_user_id)):
    conn = get_db()
    rows = conn.execute(
        """SELECT role, content, created_at FROM messages
           WHERE user_id=? AND conversation_id=? ORDER BY created_at, rowid""",
        (user_id, conversation_id)).fetchall()
    conn.close()
    return {"messages": [dict(row) for row in rows]}


@router.delete("/conversations/{conversation_id}", status_code=204)
async def clear_conversation(conversation_id: str, user_id: str = Depends(current_user_id)):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE user_id=? AND conversation_id=?", (user_id, conversation_id))
    conn.commit()
    conn.close()
