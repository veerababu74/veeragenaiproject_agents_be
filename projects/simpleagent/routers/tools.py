"""The tool library: what exists, what the agent is allowed to use, and the
user-defined HTTP tools that sit alongside the built-ins."""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from core.auth import current_user_id
from projects.simpleagent.database import get_db
from projects.simpleagent.models import BuiltinToolRequest, CustomToolRequest
from projects.simpleagent.routers.agent import agent_row
from projects.simpleagent.services.builtin_tools import BUILTIN_TOOLS
from projects.simpleagent.services.tool_builder import MAX_TOOLS_PER_AGENT, api_name

router = APIRouter(prefix="/tools", tags=["tools"])


def _missing_config(tool_type, config):
    """Which required credentials are absent. `a|b` means either satisfies it."""
    entry = BUILTIN_TOOLS.get(tool_type, {})
    missing = []
    for requirement in entry.get("requires", []):
        if not any((config or {}).get(option, "") for option in requirement.split("|")):
            missing.append(requirement)
    return missing


def _attached_ids(user_id, agent_id):
    if not agent_id:
        return set()
    conn = get_db()
    rows = conn.execute("SELECT tool_id FROM tool_links WHERE agent_id=? AND user_id=?",
                        (agent_id, user_id)).fetchall()
    conn.close()
    return {row["tool_id"] for row in rows}


@router.get("/catalog")
async def catalog():
    return {
        "tools": [
            {
                "tool_type": tool_type,
                "name": entry["name"],
                "description": entry["description"],
                "category": entry["category"],
                "config_fields": entry["config_fields"],
                "requires": entry["requires"],
                "needs_key": bool(entry["requires"]),
            }
            for tool_type, entry in BUILTIN_TOOLS.items()
        ],
        "max_per_agent": MAX_TOOLS_PER_AGENT,
    }


@router.get("")
async def list_tools(user_id: str = Depends(current_user_id)):
    agent = agent_row(user_id)
    attached = _attached_ids(user_id, agent["id"] if agent else None)
    conn = get_db()
    rows = conn.execute(
        """SELECT t.*, c.api_url, c.method FROM tools t
           LEFT JOIN custom_tools c ON c.tool_id = t.id
           WHERE t.user_id=? ORDER BY t.created_at, t.rowid""", (user_id,)).fetchall()
    conn.close()
    tools = []
    for row in rows:
        tool = dict(row)
        # The stored config can hold a credential, so only report whether one is
        # set rather than sending it back to the browser.
        config = json.loads(tool.pop("config", "{}") or "{}")
        tool["has_credentials"] = any(key in config and config[key] for key in ("api_key", "webhook_url", "token"))
        tool["config_summary"] = {key: value for key, value in config.items()
                                  if key not in ("api_key", "webhook_url", "token", "embedding_api_key")}
        tool["attached"] = tool["id"] in attached
        tools.append(tool)
    return {"tools": tools, "attached_count": len(attached), "max_per_agent": MAX_TOOLS_PER_AGENT}


@router.post("/builtin", status_code=201)
async def add_builtin_tool(request: BuiltinToolRequest, user_id: str = Depends(current_user_id)):
    entry = BUILTIN_TOOLS.get(request.tool_type)
    if not entry:
        raise HTTPException(400, f"Unknown tool type: {request.tool_type}")
    missing = _missing_config(request.tool_type, request.config)
    if missing:
        readable = ", ".join(option.replace("|", " or ") for option in missing)
        raise HTTPException(400, f"{entry['name']} needs: {readable}")

    tool_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = get_db()
    conn.execute(
        """INSERT INTO tools (id, user_id, name, description, tool_type, is_builtin, config, created_at, updated_at)
           VALUES (?,?,?,?,?,1,?,?,?)""",
        (tool_id, user_id, request.name or entry["name"], request.description or entry["description"],
         request.tool_type, json.dumps(request.config or {}), now, now))
    conn.commit()
    conn.close()
    return {"id": tool_id, "tool_type": request.tool_type, "name": request.name or entry["name"]}


@router.post("/custom", status_code=201)
async def add_custom_tool(request: CustomToolRequest, user_id: str = Depends(current_user_id)):
    if not request.api_url.startswith(("http://", "https://")):
        raise HTTPException(400, "The API URL must start with http:// or https://")

    tool_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = get_db()
    conn.execute(
        """INSERT INTO tools (id, user_id, name, description, tool_type, is_builtin, config, created_at, updated_at)
           VALUES (?,?,?,?,'custom',0,'{}',?,?)""",
        (tool_id, user_id, request.name, request.description, now, now))
    conn.execute(
        """INSERT INTO custom_tools (id, tool_id, api_url, method, headers, auth_type, auth_config,
                                     params_schema, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (str(uuid.uuid4()), tool_id, request.api_url, request.method, json.dumps(request.headers or {}),
         request.auth_type, json.dumps(request.auth_config or {}),
         json.dumps({"fields": [field.model_dump() for field in request.fields]}), now))
    conn.commit()
    conn.close()
    return {"id": tool_id, "name": request.name, "api_name": api_name(request.name)}


@router.delete("/{tool_id}", status_code=204)
async def delete_tool(tool_id: str, user_id: str = Depends(current_user_id)):
    conn = get_db()
    found = conn.execute("SELECT id FROM tools WHERE id=? AND user_id=?", (tool_id, user_id)).fetchone()
    if not found:
        conn.close()
        raise HTTPException(404, "Tool not found")
    conn.execute("DELETE FROM tools WHERE id=?", (tool_id,))
    conn.commit()
    conn.close()


@router.post("/{tool_id}/attach")
async def attach_tool(tool_id: str, user_id: str = Depends(current_user_id)):
    agent = agent_row(user_id)
    if not agent:
        raise HTTPException(400, "Create the agent before attaching tools to it")
    conn = get_db()
    if not conn.execute("SELECT id FROM tools WHERE id=? AND user_id=?", (tool_id, user_id)).fetchone():
        conn.close()
        raise HTTPException(404, "Tool not found")
    attached = conn.execute("SELECT COUNT(*) AS total FROM tool_links WHERE agent_id=?",
                            (agent["id"],)).fetchone()["total"]
    already = conn.execute("SELECT id FROM tool_links WHERE agent_id=? AND tool_id=?",
                           (agent["id"], tool_id)).fetchone()
    if not already and attached >= MAX_TOOLS_PER_AGENT:
        conn.close()
        raise HTTPException(400, f"An agent can hold at most {MAX_TOOLS_PER_AGENT} tools. Detach one first.")
    if not already:
        conn.execute(
            "INSERT INTO tool_links (id, user_id, agent_id, tool_id, created_at) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), user_id, agent["id"], tool_id, datetime.utcnow().isoformat()))
        conn.commit()
    conn.close()
    return {"attached": True, "tool_id": tool_id}


@router.delete("/{tool_id}/attach", status_code=204)
async def detach_tool(tool_id: str, user_id: str = Depends(current_user_id)):
    agent = agent_row(user_id)
    if not agent:
        raise HTTPException(404, "No agent")
    conn = get_db()
    conn.execute("DELETE FROM tool_links WHERE agent_id=? AND tool_id=? AND user_id=?",
                 (agent["id"], tool_id, user_id))
    conn.commit()
    conn.close()
