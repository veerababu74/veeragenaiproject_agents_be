"""Running one agent and narrating it as it happens.

The loop itself is small: a LangGraph cycle between a model node and a tool
node, which is the whole of ReAct. What this module adds is the commentary —
every model decision, every tool call and its result is pushed to a queue the
moment it occurs, so the browser can draw the run while it is still going
rather than showing a spinner and then a finished answer.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from projects.simpleagent.database import get_db
from projects.simpleagent.services.llm_provider import llm_for_agent
from projects.simpleagent.services.tool_builder import build_tools, describe_tools
from projects.simpleagent.services.tracing import (ANSWER, CONTEXT, ERROR, QUESTION, RunTracer)

logger = logging.getLogger("simpleagent.runner")

HISTORY_LIMIT = 12

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Appended only when the agent has tools. It is shown to the user in the trace's
# opening step, so nothing the model was told is hidden from the person reading
# the run.
TOOL_GUIDANCE = (
    "\n\nYou have tools available. Use them instead of guessing whenever a question depends on "
    "current information, on a calculation, or on the user's own documents. You may call several "
    "tools, and you may call another one after seeing what the first returned. When you have "
    "enough information, answer in your own words and say which tool the facts came from."
)


def load_agent(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM agents WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def compose_system_prompt(agent, has_tools):
    prompt = (agent["system_prompt"] or "").strip() or DEFAULT_SYSTEM_PROMPT
    description = (agent["description"] or "").strip()
    if description:
        prompt = f"{prompt}\n\nYour role: {description}"
    return prompt + TOOL_GUIDANCE if has_tools else prompt


def build_graph(user_id, agent):
    llm = llm_for_agent(user_id, agent)
    tools = build_tools(user_id, agent["id"])
    system_prompt = compose_system_prompt(agent, bool(tools))
    model = llm.bind_tools(tools) if tools else llm

    def call_model(state):
        return {"messages": [model.invoke([SystemMessage(content=system_prompt)] + state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)
    graph.add_edge(START, "agent")
    return graph.compile(), tools, system_prompt


# ── conversation memory ─────────────────────────────────────────────────────

def load_history(user_id, conversation_id):
    if not conversation_id:
        return []
    conn = get_db()
    rows = conn.execute(
        """SELECT role, content FROM messages WHERE user_id=? AND conversation_id=?
           ORDER BY created_at DESC, rowid DESC LIMIT ?""",
        (user_id, conversation_id, HISTORY_LIMIT)).fetchall()
    conn.close()
    return [HumanMessage(content=row["content"]) if row["role"] == "user" else AIMessage(content=row["content"])
            for row in reversed(rows)]


def save_turn(user_id, conversation_id, question, answer):
    if not conversation_id:
        return
    now = datetime.utcnow().isoformat()
    conn = get_db()
    conn.executemany(
        "INSERT INTO messages (id, user_id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?,?)",
        [(str(uuid.uuid4()), user_id, conversation_id, "user", question, now),
         (str(uuid.uuid4()), user_id, conversation_id, "assistant", answer, now)])
    conn.commit()
    conn.close()


# ── run bookkeeping ─────────────────────────────────────────────────────────

def _start_run(user_id, agent_id, conversation_id, question):
    run_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        """INSERT INTO runs (id, user_id, agent_id, conversation_id, question, status, created_at)
           VALUES (?,?,?,?,?,'running',?)""",
        (run_id, user_id, agent_id, conversation_id, question, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return run_id


def _finish_run(run_id, *, answer="", status="completed", error="", duration_ms=0, tracer=None):
    conn = get_db()
    conn.execute(
        """UPDATE runs SET answer=?, status=?, error_message=?, duration_ms=?,
                           tokens_in=?, tokens_out=?, rounds=?, tool_calls=? WHERE id=?""",
        (answer, status, error, duration_ms,
         tracer.tokens_in if tracer else 0, tracer.tokens_out if tracer else 0,
         tracer.rounds if tracer else 0, tracer.tool_calls if tracer else 0, run_id))
    conn.commit()
    conn.close()


def _final_text(state):
    """The answer is the last AI message carrying real text — not the whole
    transcript, which would echo the question and raw tool output back."""
    for message in reversed((state or {}).get("messages", [])):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content.strip()
    return ""


# ── streaming ───────────────────────────────────────────────────────────────

async def stream_run(user_id, message, conversation_id):
    agent = load_agent(user_id)
    if not agent:
        yield {"type": "error", "error": "Create an agent before chatting with it."}
        return

    run_id = _start_run(user_id, agent["id"], conversation_id, message)
    started = time.time()
    queue = asyncio.Queue()
    tracer = RunTracer(run_id, user_id, queue)

    yield {"type": "start", "run_id": run_id}

    try:
        graph, tools, system_prompt = build_graph(user_id, agent)
    except Exception as error:
        tracer.add(ERROR, content=str(error))
        tracer.flush()
        _finish_run(run_id, status="error", error=str(error),
                    duration_ms=int((time.time() - started) * 1000), tracer=tracer)
        yield {"type": "error", "error": str(error)}
        return

    # Everything the model is about to be given, recorded before it is given —
    # the prompt, the role, and the exact tool descriptions it will choose from.
    tool_specs = describe_tools(tools)
    tracer.add(CONTEXT, content=system_prompt, round_number=0, data={
        "agent": {"name": agent["name"], "description": agent["description"],
                  "provider": agent["provider"], "model": agent["model"],
                  "temperature": agent["temperature"]},
        "tools": tool_specs,
        "tool_count": len(tool_specs),
    })
    tracer.add(QUESTION, content=message, round_number=0)
    while not queue.empty():
        yield queue.get_nowait()

    history = load_history(user_id, conversation_id)
    chunks, final_state = [], {}

    async def run():
        nonlocal final_state
        async for event in graph.astream_events(
            {"messages": history + [HumanMessage(content=message)]},
            version="v2", config={"callbacks": [tracer]},
        ):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                text = getattr(event["data"].get("chunk"), "content", "")
                if isinstance(text, str) and text:
                    chunks.append(text)
                    queue.put_nowait({"type": "token", "text": text})
            elif kind == "on_chain_end" and event.get("name") == "LangGraph" and not event.get("parent_ids"):
                final_state = event["data"].get("output") or {}

    task = asyncio.create_task(run())
    try:
        while True:
            drain = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({drain, task}, return_when=asyncio.FIRST_COMPLETED)
            if drain in done:
                yield drain.result()
                continue
            drain.cancel()  # nothing was taken from the queue, so nothing is lost
            while not queue.empty():
                yield queue.get_nowait()
            await task  # re-raises whatever the run failed with
            break
    except Exception as error:
        task.cancel()
        duration = int((time.time() - started) * 1000)
        tracer.add(ERROR, content=str(error), duration_ms=duration)
        tracer.flush()
        _finish_run(run_id, status="error", error=str(error), duration_ms=duration, tracer=tracer)
        yield {"type": "error", "error": str(error)}
        return

    answer = _final_text(final_state) or "".join(chunks).strip()
    duration = int((time.time() - started) * 1000)
    tracer.add(ANSWER, content=answer, duration_ms=duration, data={
        "tokens": {"in": tracer.tokens_in, "out": tracer.tokens_out},
        "rounds": tracer.rounds, "tool_calls": tracer.tool_calls,
    })
    while not queue.empty():  # the answer step was queued after the last drain
        yield queue.get_nowait()

    tracer.flush()
    _finish_run(run_id, answer=answer, duration_ms=duration, tracer=tracer)
    save_turn(user_id, conversation_id, message, answer)

    yield {
        "type": "done", "run_id": run_id, "answer": answer, "duration_ms": duration,
        "rounds": tracer.rounds, "tool_calls": tracer.tool_calls,
        "tokens": {"in": tracer.tokens_in, "out": tracer.tokens_out},
    }


def run_summary(steps):
    """Counts the UI puts above a finished run."""
    calls = [step for step in steps if step["step_type"] == "tool_call"]
    return {
        "steps": len(steps),
        "rounds": max([step["round"] for step in steps] or [0]),
        "tool_calls": len(calls),
        "tools_used": list(dict.fromkeys(step["tool_name"] for step in calls)),
        "errors": len([step for step in steps if step["step_type"] in ("tool_error", "error")]),
        "tool_sequence": [{"order": step["tool_order"], "tool": step["tool_name"], "round": step["round"]}
                          for step in calls],
    }
