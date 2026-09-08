"""Marketing Copilot's schema, and the retention sweep the platform runs hourly.

Two kinds of table live here, and keeping them in one database is deliberate.

The *business* tables — campaigns, documents, compliance rules — are what the
copilot answers questions about. The *observability* tables — messages, feedback,
eval_runs — are what let anyone answer questions about the copilot. Because they
share a database, "show me every thumbs-down answer from this week and the chunks
it cited" is one join rather than a data-engineering project. That query is how
retrieval problems are actually found, so the schema is built to make it cheap.

Everything a user creates is swept past the platform's retention window. The
seeded demo corpus is not: it is re-created at startup and belongs to no user.
"""

import json
import logging
import uuid

from core.config import settings
from core.database import Database

logger = logging.getLogger("marketingcopilot.database")

SCHEMA = """
-- ── what the copilot answers questions about ────────────────────────────────

CREATE TABLE IF NOT EXISTS campaigns (
    id             TEXT PRIMARY KEY,
    workspace_id   TEXT NOT NULL,
    name           TEXT NOT NULL,
    channel        TEXT NOT NULL,
    segment        TEXT NOT NULL,
    quarter        TEXT NOT NULL,
    spend          REAL NOT NULL,
    impressions    INTEGER NOT NULL,
    clicks         INTEGER NOT NULL,
    conversions    INTEGER NOT NULL,
    pipeline_value REAL NOT NULL,
    created_at     TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_campaigns_lookup
    ON campaigns (workspace_id, quarter, channel, segment);

CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    title        TEXT NOT NULL,
    doc_type     TEXT NOT NULL,
    channel      TEXT DEFAULT '',
    segment      TEXT DEFAULT '',
    quarter      TEXT DEFAULT '',
    body         TEXT NOT NULL,
    -- Re-uploading an unchanged document must be a no-op. Without this the
    -- corpus silently fills with duplicate chunks, which degrades retrieval in
    -- a way that produces no error anywhere.
    content_hash TEXT NOT NULL,
    chunk_count  INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT (datetime('now')),
    UNIQUE (workspace_id, title)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    section      TEXT DEFAULT '',
    text         TEXT NOT NULL,
    -- Only populated when the local retriever is in use; with Pinecone the
    -- vectors live there and this stays null.
    embedding    TEXT,
    created_at   TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_workspace ON chunks (workspace_id);

CREATE TABLE IF NOT EXISTS compliance_rules (
    id           TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    code         TEXT NOT NULL,
    rule         TEXT NOT NULL,
    severity     TEXT NOT NULL DEFAULT 'high',
    pattern      TEXT NOT NULL DEFAULT '',
    created_at   TEXT DEFAULT (datetime('now'))
);

-- ── the user's own setup ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS settings_rows (
    user_id      TEXT PRIMARY KEY,
    provider     TEXT NOT NULL DEFAULT 'openai',
    chat_model   TEXT NOT NULL DEFAULT 'gpt-4o-mini',
    embed_model  TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    api_key      TEXT NOT NULL DEFAULT '',
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now'))
);

-- ── conversation and observability ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT DEFAULT 'New conversation',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    route           TEXT DEFAULT '',
    citations       TEXT DEFAULT '[]',
    steps           TEXT DEFAULT '[]',
    sql_query       TEXT DEFAULT '',
    attempts        INTEGER DEFAULT 0,
    prompt_tokens   INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    latency_ms      INTEGER DEFAULT 0,
    top_score       REAL DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages (user_id, created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id         TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    rating     INTEGER NOT NULL,
    reason     TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    dataset    TEXT NOT NULL,
    metrics    TEXT NOT NULL DEFAULT '{}',
    cases      TEXT NOT NULL DEFAULT '[]',
    created_at TEXT DEFAULT (datetime('now'))
);
"""

database = Database("marketingcopilot", SCHEMA)

# The seeded corpus belongs to nobody, so every user shares one read-only
# workspace. A real deployment would key this by tenant; the shape is the same.
DEMO_WORKSPACE = "demo"


def init_db() -> None:
    database.initialize()
    from projects.marketingcopilot.corpus import seed_workspace

    seed_workspace()


def cleanup_expired_data():
    """Sweep user data past the retention window. The demo corpus is exempt --
    it is content, not user state, and re-seeding it hourly would be pointless
    churn."""
    conn = database.connect()
    hours = settings.retention_hours
    cursor = conn.execute(
        "DELETE FROM messages WHERE created_at < datetime('now', ?)", (f"-{hours} hours",))
    removed = cursor.rowcount
    for table in ("feedback", "eval_runs", "conversations"):
        conn.execute(
            f"DELETE FROM {table} WHERE created_at < datetime('now', ?)", (f"-{hours} hours",))
    conn.commit()
    conn.close()
    return {"messages_removed": removed}


# ── small helpers, so routers do not open connections by hand ────────────────

def new_id() -> str:
    return uuid.uuid4().hex


def query(sql: str, params: tuple = ()) -> list[dict]:
    conn = database.connect()
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def execute(sql: str, params: tuple = ()) -> None:
    conn = database.connect()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def get_settings_row(user_id: str) -> dict | None:
    rows = query("SELECT * FROM settings_rows WHERE user_id = ?", (user_id,))
    return rows[0] if rows else None


def save_settings_row(user_id: str, provider: str, chat_model: str,
                      embed_model: str, api_key: str) -> None:
    execute(
        """INSERT INTO settings_rows (user_id, provider, chat_model, embed_model, api_key)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(user_id) DO UPDATE SET
             provider = excluded.provider, chat_model = excluded.chat_model,
             embed_model = excluded.embed_model, api_key = excluded.api_key,
             updated_at = datetime('now')""",
        (user_id, provider, chat_model, embed_model, api_key),
    )


def record_message(**fields) -> str:
    """Insert one message and return its id.

    Every observability column is written here rather than at a dozen call
    sites, so a message can never reach the table without its route, citations
    and latency attached. A row missing those is a row you cannot debug later.
    """
    message_id = fields.pop("id", None) or new_id()
    columns = {
        "id": message_id,
        "conversation_id": fields["conversation_id"],
        "user_id": fields["user_id"],
        "role": fields["role"],
        "content": fields["content"],
        "route": fields.get("route", ""),
        "citations": json.dumps(fields.get("citations", [])),
        "steps": json.dumps(fields.get("steps", [])),
        "sql_query": fields.get("sql_query", "") or "",
        "attempts": fields.get("attempts", 0),
        "prompt_tokens": fields.get("prompt_tokens", 0),
        "completion_tokens": fields.get("completion_tokens", 0),
        "latency_ms": fields.get("latency_ms", 0),
        "top_score": fields.get("top_score", 0.0),
    }
    placeholders = ", ".join("?" for _ in columns)
    execute(f"INSERT INTO messages ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(columns.values()))
    return message_id
