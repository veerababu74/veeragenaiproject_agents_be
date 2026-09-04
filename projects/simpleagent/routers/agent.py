"""The agent and the provider keys it runs on.

One agent per user, deliberately: this project is about watching a single agent
reason, so there is nothing to select between and no way to end up chatting with
the wrong one.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from core.auth import current_user_id
from projects.simpleagent.database import get_db
from projects.simpleagent.models import AgentRequest, ProviderKeyRequest
from projects.simpleagent.services.llm_provider import PROVIDER_CATALOG, get_key, save_key

router = APIRouter(tags=["agent"])


def agent_row(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM agents WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def linked_tools(user_id, agent_id):
    conn = get_db()
    rows = conn.execute(
        """SELECT t.id, t.name, t.description, t.tool_type, t.is_builtin
           FROM tool_links l JOIN tools t ON t.id = l.tool_id
           WHERE l.agent_id=? AND l.user_id=? ORDER BY l.created_at, t.rowid""",
        (agent_id, user_id)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def agent_payload(user_id):
    agent = agent_row(user_id)
    if not agent:
        return None
    agent["tools"] = linked_tools(user_id, agent["id"])
    agent["has_key"] = bool(get_key(user_id, agent["provider"]))
    return agent


@router.get("/providers")
async def providers():
    return {
        "providers": [
            {"id": key, "label": value["label"], "models": value["models"], "key_hint": value["key_hint"]}
            for key, value in PROVIDER_CATALOG.items()
        ]
    }


@router.get("/agent")
async def get_agent(user_id: str = Depends(current_user_id)):
    return agent_payload(user_id)


@router.post("/agent")
async def save_agent(request: AgentRequest, user_id: str = Depends(current_user_id)):
    """Create the agent, or update it if one already exists. The API key travels
    with the form because that is the moment the user has it to hand, but it is
    stored per provider rather than on the agent, so switching model keeps it."""
    if request.api_key.strip():
        save_key(user_id, request.provider, request.api_key)

    now = datetime.utcnow().isoformat()
    existing = agent_row(user_id)
    conn = get_db()
    if existing:
        conn.execute(
            """UPDATE agents SET name=?, description=?, system_prompt=?, provider=?, model=?,
                                 temperature=?, max_tokens=?, updated_at=? WHERE id=?""",
            (request.name, request.description, request.system_prompt, request.provider, request.model,
             request.temperature, request.max_tokens, now, existing["id"]))
    else:
        conn.execute(
            """INSERT INTO agents (id, user_id, name, description, system_prompt, provider, model,
                                   temperature, max_tokens, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), user_id, request.name, request.description, request.system_prompt,
             request.provider, request.model, request.temperature, request.max_tokens, now, now))
    conn.commit()
    conn.close()
    return agent_payload(user_id)


@router.delete("/agent", status_code=204)
async def delete_agent(user_id: str = Depends(current_user_id)):
    conn = get_db()
    conn.execute("DELETE FROM agents WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


@router.get("/keys")
async def list_keys(user_id: str = Depends(current_user_id)):
    """Which providers have a key saved. The keys themselves are never returned."""
    conn = get_db()
    rows = conn.execute("SELECT provider, updated_at FROM provider_keys WHERE user_id=?", (user_id,)).fetchall()
    conn.close()
    saved = {row["provider"]: row["updated_at"] for row in rows}
    return {
        "keys": [
            {"provider": key, "label": value["label"], "saved": key in saved, "updated_at": saved.get(key)}
            for key, value in PROVIDER_CATALOG.items()
        ]
    }


@router.post("/keys")
async def upsert_key(request: ProviderKeyRequest, user_id: str = Depends(current_user_id)):
    save_key(user_id, request.provider, request.api_key)
    return {"provider": request.provider, "saved": True}


@router.delete("/keys/{provider}", status_code=204)
async def delete_key(provider: str, user_id: str = Depends(current_user_id)):
    if provider not in PROVIDER_CATALOG:
        raise HTTPException(404, "Unknown provider")
    conn = get_db()
    conn.execute("DELETE FROM provider_keys WHERE user_id=? AND provider=?", (user_id, provider))
    conn.commit()
    conn.close()
