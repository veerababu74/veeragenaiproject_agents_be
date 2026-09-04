"""Ready-made agents, and applying one in a single call.

Applying an example configures the agent and attaches every tool in it that
works without a credential. Tools that need an API key are reported back rather
than created half-configured, so the user is told what is still missing instead
of discovering it when the agent silently fails to use them.
"""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth import current_user_id
from projects.simpleagent.database import get_db
from projects.simpleagent.models import Provider
from projects.simpleagent.routers.agent import agent_payload, agent_row
from projects.simpleagent.services.builtin_tools import BUILTIN_TOOLS
from projects.simpleagent.services.examples import EXAMPLES, example_by_id
from projects.simpleagent.services.llm_provider import get_key, save_key
from projects.simpleagent.services.tool_builder import MAX_TOOLS_PER_AGENT

router = APIRouter(prefix="/examples", tags=["examples"])


class ApplyExampleRequest(BaseModel):
    provider: Provider = "openai"
    model: str = Field(min_length=1, max_length=120)
    api_key: str = Field(default="", max_length=400)


@router.get("")
async def list_examples():
    return {
        "examples": [
            {
                **example,
                "tools": [
                    {"tool_type": tool_type,
                     "name": BUILTIN_TOOLS[tool_type]["name"],
                     "needs_key": bool(BUILTIN_TOOLS[tool_type]["requires"])}
                    for tool_type in example["tools"] if tool_type in BUILTIN_TOOLS
                ],
            }
            for example in EXAMPLES
        ]
    }


@router.post("/{example_id}/apply")
async def apply_example(example_id: str, request: ApplyExampleRequest,
                        user_id: str = Depends(current_user_id)):
    example = example_by_id(example_id)
    if not example:
        raise HTTPException(404, "Unknown example")
    if request.api_key.strip():
        save_key(user_id, request.provider, request.api_key)
    if not get_key(user_id, request.provider):
        raise HTTPException(400, "Add an API key for this provider first")

    now = datetime.utcnow().isoformat()
    existing = agent_row(user_id)
    conn = get_db()
    if existing:
        agent_id = existing["id"]
        conn.execute(
            """UPDATE agents SET name=?, description=?, system_prompt=?, provider=?, model=?, updated_at=?
               WHERE id=?""",
            (example["title"], example["description"], example["system_prompt"],
             request.provider, request.model, now, agent_id))
        # An example is a complete configuration, so it replaces the previous
        # tool set rather than merging into it.
        conn.execute("DELETE FROM tool_links WHERE agent_id=?", (agent_id,))
    else:
        agent_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO agents (id, user_id, name, description, system_prompt, provider, model,
                                   temperature, max_tokens, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,0.3,2048,?,?)""",
            (agent_id, user_id, example["title"], example["description"], example["system_prompt"],
             request.provider, request.model, now, now))

    attached, needs_setup = [], []
    for tool_type in example["tools"][:MAX_TOOLS_PER_AGENT]:
        entry = BUILTIN_TOOLS.get(tool_type)
        if not entry:
            continue
        if entry["requires"]:
            needs_setup.append({"tool_type": tool_type, "name": entry["name"],
                                "requires": entry["requires"]})
            continue
        tool_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO tools (id, user_id, name, description, tool_type, is_builtin, config, created_at, updated_at)
               VALUES (?,?,?,?,?,1,?,?,?)""",
            (tool_id, user_id, entry["name"], entry["description"], tool_type, json.dumps({}), now, now))
        conn.execute(
            "INSERT INTO tool_links (id, user_id, agent_id, tool_id, created_at) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), user_id, agent_id, tool_id, now))
        attached.append(entry["name"])
    conn.commit()
    conn.close()

    return {
        "agent": agent_payload(user_id),
        "attached": attached,
        "needs_setup": needs_setup,
        "sample_questions": example["sample_questions"],
    }
