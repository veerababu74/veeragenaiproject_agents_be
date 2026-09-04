"""SimpleAgent's schema and its retention sweep.

Everything a user creates here is temporary by design: the point of the project
is to watch one run happen, not to host a durable workspace. Every table carries
created_at and is swept once past the platform's retention window.
"""

import logging

from core.config import settings
from core.database import Database

logger = logging.getLogger("simpleagent.database")

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    system_prompt TEXT DEFAULT '',
    provider TEXT NOT NULL DEFAULT 'openai',
    model TEXT NOT NULL DEFAULT 'gpt-4o-mini',
    temperature REAL DEFAULT 0.3,
    max_tokens INTEGER DEFAULT 2048,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS provider_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    api_key TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, provider)
);

CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    tool_type TEXT NOT NULL,
    is_builtin INTEGER DEFAULT 1,
    config TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS custom_tools (
    id TEXT PRIMARY KEY,
    tool_id TEXT NOT NULL UNIQUE,
    api_url TEXT NOT NULL,
    method TEXT DEFAULT 'GET',
    headers TEXT DEFAULT '{}',
    auth_type TEXT DEFAULT 'none',
    auth_config TEXT DEFAULT '{}',
    params_schema TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tool_links (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(agent_id, tool_id),
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER DEFAULT 0,
    remote_path TEXT DEFAULT '',
    embedding_model TEXT DEFAULT '',
    chunk_strategy TEXT DEFAULT 'recursive',
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 150,
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'processing',
    error_message TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    conversation_id TEXT DEFAULT '',
    question TEXT DEFAULT '',
    answer TEXT DEFAULT '',
    status TEXT DEFAULT 'running',
    error_message TEXT DEFAULT '',
    duration_ms INTEGER DEFAULT 0,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    rounds INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS run_steps (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    round INTEGER DEFAULT 0,
    step_type TEXT NOT NULL,
    tool_name TEXT DEFAULT '',
    tool_order INTEGER DEFAULT 0,
    content TEXT DEFAULT '',
    data TEXT DEFAULT '{}',
    duration_ms INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tools_user ON tools(user_id);
CREATE INDEX IF NOT EXISTS idx_links_agent ON tool_links(agent_id);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_user ON runs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_steps_run ON run_steps(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(user_id, conversation_id, created_at);
"""

# Children before the rows they reference, so a delete never strands a row.
RETAINED_TABLES = ("run_steps", "runs", "messages", "documents", "tool_links",
                   "custom_tools", "tools", "provider_keys", "agents")

database = Database("simpleagent", SCHEMA)


def get_db():
    return database.connect()


def init_db() -> None:
    database.initialize()


def cleanup_expired_data() -> int:
    """Remove the external copies first, then the rows that point at them, so
    nothing is left stranded in a service we can no longer address."""
    from core.storage import BucketError
    from core.vectors import VectorStoreError
    from projects.simpleagent.resources import bucket, vectors

    hours = settings.retention_hours
    conn = get_db()
    expiring = [dict(row) for row in conn.execute(
        "SELECT id, user_id, chunk_count, remote_path, status FROM documents "
        "WHERE created_at < datetime('now', ?)", (f"-{hours} hours",))]
    conn.close()

    for document in expiring:
        if document["status"] != "ready":
            continue
        try:
            vectors.delete_document(document["user_id"], document["id"], document["chunk_count"] or 0)
        except VectorStoreError:
            logger.warning("Could not delete vectors for expired document %s", document["id"])
        if document["remote_path"]:
            try:
                bucket.delete(document["remote_path"])
            except BucketError:
                logger.warning("Could not delete the stored file for expired document %s", document["id"])

    database.purge_older_than(hours, RETAINED_TABLES)
    return len(expiring)
