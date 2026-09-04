"""The trace: an ordered record of what the agent thought, chose, ran and answered.

Implemented as a LangChain callback handler rather than by instrumenting the
loop by hand, because the callbacks fire from inside the graph and therefore see
every model decision and every tool invocation in the order they really happen.

Two numbers make the timeline readable, and they are the reason this exists as
its own module rather than a list of log lines:

  round — one pass of the think/act loop. The model is called, it either answers
          or picks tools, the tools run, and the model is called again with what
          they returned. Round 2 existing at all is the proof that the agent
          reacted to a tool result rather than planning everything up front.

  order — the position of a tool call across the whole run, so "it searched, then
          read the page it found, then did the arithmetic" is legible as a
          sequence instead of three unrelated events.

Events are pushed to a queue as they happen (the live view) and written to
SQLite once at the end (the replay), because one transaction keeps the sequence
contiguous and a run is short enough that nothing is lost by waiting.
"""

import json
import time
import uuid
from datetime import datetime

from langchain_core.callbacks import AsyncCallbackHandler

QUESTION = "question"
CONTEXT = "context"
THINK = "think"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
TOOL_ERROR = "tool_error"
ANSWER = "answer"
ERROR = "error"

MAX_TEXT = 4000


def clip(value, limit=MAX_TEXT):
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"… [truncated, {len(text)} characters]"


class RunTracer(AsyncCallbackHandler):
    def __init__(self, run_id, user_id, queue=None):
        self.run_id = run_id
        self.user_id = user_id
        self.queue = queue
        self.events = []
        self._seq = 0
        self._round = 0
        self._tool_order = 0
        self._pending = {}       # langchain run_id -> (started_at, tool name)
        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def rounds(self):
        return self._round

    @property
    def tool_calls(self):
        return self._tool_order

    def add(self, step_type, *, tool_name="", tool_order=0, content="", data=None, duration_ms=0, round_number=None):
        self._seq += 1
        event = {
            "seq": self._seq,
            "round": self._round if round_number is None else round_number,
            "step_type": step_type,
            "tool_name": tool_name,
            "tool_order": tool_order,
            "content": clip(content),
            "data": data or {},
            "duration_ms": duration_ms,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.events.append(event)
        if self.queue is not None:
            self.queue.put_nowait({"type": "step", **event})
        return event

    # ── model ───────────────────────────────────────────────────────────────

    async def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._round += 1
        self._pending[run_id] = (time.monotonic(), "")

    async def on_llm_end(self, response, *, run_id=None, **kwargs):
        started, _ = self._pending.pop(run_id, (time.monotonic(), ""))
        duration = int((time.monotonic() - started) * 1000)

        message = None
        for generations in getattr(response, "generations", []) or []:
            for generation in generations:
                message = getattr(generation, "message", None) or message

        usage = getattr(message, "usage_metadata", None) or {}
        self.tokens_in += usage.get("input_tokens") or 0
        self.tokens_out += usage.get("output_tokens") or 0

        text = getattr(message, "content", "")
        if not isinstance(text, str):
            text = clip(text)
        tool_calls = getattr(message, "tool_calls", None) or []

        # A decision to answer is recorded as the answer, not as thinking, so the
        # same text never appears twice in the timeline.
        if not tool_calls:
            return
        self.add(THINK, content=text, duration_ms=duration, data={
            "chose": [call.get("name") for call in tool_calls],
            "parallel": len(tool_calls) > 1,
            "arguments": {call.get("name"): call.get("args", {}) for call in tool_calls},
            "tokens": {"in": usage.get("input_tokens", 0), "out": usage.get("output_tokens", 0)},
        })

    # ── tools ───────────────────────────────────────────────────────────────

    async def on_tool_start(self, serialized, input_str, *, run_id=None, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or "tool"
        self._tool_order += 1
        self._pending[run_id] = (time.monotonic(), name, self._tool_order)
        self.add(TOOL_CALL, tool_name=name, tool_order=self._tool_order,
                 content=clip(inputs if inputs is not None else input_str, 1200),
                 data={"arguments": inputs if isinstance(inputs, dict) else {"input": clip(input_str, 1200)}})

    async def on_tool_end(self, output, *, run_id=None, **kwargs):
        started, name, order = self._pending.pop(run_id, (time.monotonic(), "tool", 0))
        content = getattr(output, "content", output)
        self.add(TOOL_RESULT, tool_name=name, tool_order=order, content=clip(content),
                 duration_ms=int((time.monotonic() - started) * 1000))

    async def on_tool_error(self, error, *, run_id=None, **kwargs):
        started, name, order = self._pending.pop(run_id, (time.monotonic(), "tool", 0))
        self.add(TOOL_ERROR, tool_name=name, tool_order=order, content=str(error),
                 duration_ms=int((time.monotonic() - started) * 1000))

    # ── persistence ─────────────────────────────────────────────────────────

    def flush(self):
        if not self.events:
            return
        from projects.simpleagent.database import get_db

        rows = [(str(uuid.uuid4()), self.run_id, self.user_id, event["seq"], event["round"], event["step_type"],
                 event["tool_name"], event["tool_order"], event["content"],
                 json.dumps(event["data"], default=str), event["duration_ms"], event["created_at"])
                for event in self.events]
        conn = get_db()
        conn.executemany(
            """INSERT INTO run_steps (id, run_id, user_id, seq, round, step_type, tool_name, tool_order,
                                      content, data, duration_ms, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        conn.commit()
        conn.close()
        self.events = []


def load_steps(user_id, run_id):
    from projects.simpleagent.database import get_db

    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM run_steps WHERE user_id=? AND run_id=? ORDER BY seq", (user_id, run_id)).fetchall()
    conn.close()
    steps = []
    for row in rows:
        step = dict(row)
        try:
            step["data"] = json.loads(step["data"] or "{}")
        except ValueError:
            step["data"] = {}
        steps.append(step)
    return steps
