"""Turning stored tool rows into the tools an agent actually receives.

Three kinds resolve here: built-ins from the catalogue, the document-search
retriever, and user-defined HTTP tools whose typed fields become the arguments
the model fills in.
"""

import json
import logging
import re
from urllib.parse import quote

import aiohttp
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger("simpleagent.tools")

MAX_TOOLS_PER_AGENT = 10

FIELD_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}


def api_name(name: str, fallback: str = "custom_tool") -> str:
    """Providers accept only [a-zA-Z0-9_-] in a tool name, but people type
    things like "My Weather API", so normalise instead of letting the provider
    reject the whole request."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip().lower()).strip("_")
    return slug[:60] or fallback


# ── custom HTTP tools ───────────────────────────────────────────────────────

def _args_model(fields, model_name):
    definitions = {}
    for field in fields or []:
        key = field.get("name", "")
        if not key.isidentifier():
            continue  # not addressable as a python keyword argument
        annotation = FIELD_TYPES.get(field.get("type", "string"), str)
        description = field.get("description", "") or ""
        if field.get("required", True):
            definitions[key] = (annotation, Field(description=description))
        else:
            definitions[key] = (annotation, Field(default=None, description=description))
    safe = re.sub(r"\W|^(?=\d)", "_", model_name) or "ToolInput"
    return create_model(safe, **definitions) if definitions else create_model(safe)


async def _call_api(spec, arguments):
    method = (spec.get("method") or "GET").upper()
    headers = dict(spec.get("headers") or {})
    auth_type = spec.get("auth_type", "none")
    auth = spec.get("auth_config") or {}
    if auth_type == "bearer":
        headers["Authorization"] = f"Bearer {auth.get('token', '')}"
    elif auth_type == "api_key":
        headers[auth.get("header_name") or "X-API-Key"] = auth.get("api_key", "")

    # A {placeholder} in the URL is filled from the arguments and then removed,
    # so the same value is not also sent as a query parameter or body field.
    url = spec["api_url"]
    arguments = {key: value for key, value in (arguments or {}).items() if value is not None}
    for key in list(arguments):
        token = "{" + key + "}"
        if token in url:
            url = url.replace(token, quote(str(arguments.pop(key)), safe=""))

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            kwargs = {"headers": headers}
            if method in {"GET", "DELETE"}:
                kwargs["params"] = {key: str(value) for key, value in arguments.items()}
            else:
                kwargs["json"] = arguments
                kwargs["headers"] = {**headers, "Content-Type": "application/json"}
            async with session.request(method, url, **kwargs) as response:
                body = await response.text()
                if response.status >= 400:
                    return json.dumps({"error": f"HTTP {response.status}", "body": body[:800]})
                try:
                    return json.dumps(json.loads(body))[:6000]
                except ValueError:
                    return body[:6000]
    except Exception as error:
        return json.dumps({"error": str(error)})


def create_custom_tool(row):
    spec = {
        "api_url": row["api_url"],
        "method": row["method"],
        "headers": json.loads(row["headers"] or "{}"),
        "auth_type": row["auth_type"],
        "auth_config": json.loads(row["auth_config"] or "{}"),
    }
    fields = json.loads(row["params_schema"] or "{}").get("fields", [])
    name = api_name(row["name"])

    async def _run(**arguments):
        return await _call_api(spec, arguments)

    return StructuredTool.from_function(
        coroutine=_run, name=name, args_schema=_args_model(fields, f"{name}_input"),
        description=row["description"] or f"Call the {row['name']} API.")


# ── document search ─────────────────────────────────────────────────────────

class DocumentSearchInput(BaseModel):
    query: str = Field(description="What to look for in the uploaded documents")


def _embedding_model_for(user_id):
    """The model the user's documents were embedded with. A query vector is only
    comparable to document vectors from the same model, and the upload form lets
    the user choose, so defaulting here would silently mismatch."""
    from projects.simpleagent.database import get_db
    from core.embeddings import DEFAULT_EMBEDDING_MODEL

    conn = get_db()
    row = conn.execute(
        """SELECT embedding_model FROM documents
           WHERE user_id=? AND status='ready' AND embedding_model != ''
           ORDER BY created_at DESC, rowid DESC LIMIT 1""", (user_id,)).fetchone()
    conn.close()
    return (row["embedding_model"] if row else "") or DEFAULT_EMBEDDING_MODEL


def create_document_search_tool(user_id, config):
    from core.embeddings import EmbeddingError, embed_texts
    from core.vectors import VectorStoreError
    from projects.simpleagent.resources import vectors
    from projects.simpleagent.services.llm_provider import get_key

    top_k = max(1, min(int(config.get("top_k") or 5), 10))

    async def _run(query: str):
        api_key = config.get("embedding_api_key") or get_key(user_id, "google")
        if not api_key:
            return ("No Google Gemini API key is available for embeddings, so the uploaded documents "
                    "cannot be searched. Add a Gemini key under Keys.")
        try:
            vector = embed_texts(api_key, _embedding_model_for(user_id), [query], "RETRIEVAL_QUERY")[0]
            matches = vectors.query(user_id, vector, top_k)
        except (EmbeddingError, VectorStoreError) as error:
            return f"Document search failed: {error}"
        if not matches:
            return "No relevant passages were found in the uploaded documents."
        return "\n\n".join(
            f"[{match.get('filename', 'document')} · chunk {match.get('position', 0) + 1} · "
            f"score {match.get('score', 0):.2f}]\n{match.get('text', '')}"
            for match in matches)

    return StructuredTool.from_function(
        coroutine=_run, name="document_search", args_schema=DocumentSearchInput,
        description=("Search the user's own uploaded documents for relevant passages. "
                     "Use whenever the question refers to their files, reports or data."))


# ── assembly ────────────────────────────────────────────────────────────────

def build_tools(user_id, agent_id):
    """Every tool linked to this agent, in the order they were attached.

    One malformed tool must not take the agent down with it, so a tool that
    fails to build is skipped and logged and the agent runs with the rest.
    """
    from projects.simpleagent.database import get_db
    from projects.simpleagent.services.builtin_tools import create_builtin

    conn = get_db()
    rows = conn.execute(
        """SELECT t.*, c.api_url, c.method, c.headers, c.auth_type, c.auth_config, c.params_schema
           FROM tool_links l
           JOIN tools t ON t.id = l.tool_id
           LEFT JOIN custom_tools c ON c.tool_id = t.id
           WHERE l.agent_id = ? AND l.user_id = ?
           ORDER BY l.created_at, t.rowid""",
        (agent_id, user_id)).fetchall()
    conn.close()

    tools, seen = [], set()
    for row in rows:
        try:
            config = json.loads(row["config"] or "{}")
            if row["tool_type"] == "custom":
                tool = create_custom_tool(row)
            elif row["tool_type"] == "document_search":
                tool = create_document_search_tool(user_id, config)
            else:
                tool = create_builtin(row["tool_type"], config)
        except Exception:
            logger.exception("Skipping tool %s (%s) that failed to build", row["name"], row["tool_type"])
            continue
        # Two tools with the same resolved name would be ambiguous to the model.
        if tool and tool.name not in seen:
            seen.add(tool.name)
            tools.append(tool)
    return tools


def describe_tools(tools):
    """Exactly what the model is told about each tool — name, description and
    argument schema. The UI shows this so the tool choice can be read against
    the information the choice was made from."""
    described = []
    for tool in tools:
        schema = {}
        try:
            schema = (tool.args_schema.model_json_schema() or {}).get("properties", {}) \
                if tool.args_schema else {}
        except Exception:
            schema = {}
        described.append({
            "name": tool.name,
            "description": tool.description,
            "arguments": [
                {"name": key, "type": value.get("type", "string"), "description": value.get("description", "")}
                for key, value in schema.items()
            ],
        })
    return described
