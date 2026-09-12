"""Chunking, embedding and search.

Three decisions here are worth more than the code around them.

**Chunk on structure, not on length.** These documents are briefs, playbooks and
retros with real headings, and a heading is the unit of meaning: the "Audience"
section of one brief must never be merged with the "Budget" of the next. Fixed
character windows would do exactly that. Where a section is genuinely long it
falls back to a windowed split, so the strategy degrades rather than breaks.

**Search is hybrid.** Dense vectors are weak at exact tokens -- quarter labels
like `2024-Q3`, channel names, product terms -- and this corpus is full of them.
A lexical score is fused with the dense one by reciprocal rank fusion, which
combines *rankings* rather than scores and so does not require two incomparable
scales to be comparable.

**Metadata is filtered, not embedded.** Asking for Q3 LinkedIn and getting the Q1
brief is the single most common retrieval failure in this domain, because two
briefs are semantically near-identical and the quarter is a detail the embedding
barely encodes. Anything that would be a SQL `WHERE` clause is metadata, and it
is applied as a filter before similarity is consulted at all.
"""

import json
import logging
import math
import re

from core.config import project_value, settings
from projects.marketingcopilot.providers import build_embeddings
from projects.marketingcopilot.database import (
    DEMO_WORKSPACE, database, new_id, query,
)

logger = logging.getLogger("marketingcopilot.retrieval")

SLUG = "marketingcopilot"
MAX_SECTION_CHARS = 1200
WINDOW_OVERLAP = 150

_HEADING = re.compile(r"^##\s+(.+)$", re.M)
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")

# Marketing writing is full of words that carry no retrieval signal. Dropping
# them stops the lexical half of the hybrid from matching on filler.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "our", "was", "were", "are",
    "have", "has", "had", "what", "which", "when", "how", "why", "did", "does", "not",
    "you", "your", "they", "them", "their", "than", "then", "into", "over", "under",
    "about", "would", "should", "could", "will", "can", "any", "all", "one", "two",
}


# ── chunking ─────────────────────────────────────────────────────────────────

def chunk_document(body: str) -> list[dict]:
    """Split on `##` headings, carrying the heading into the chunk text.

    The heading travels with the body because it is most of what makes the chunk
    findable: "Audience" alone is meaningless, and "Q3 brief / Audience" is not.
    """
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING.finditer(body))

    if not matches:
        sections.append(("", body.strip()))
    else:
        preamble = body[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            sections.append((match.group(1).strip(), body[match.end():end].strip()))

    chunks: list[dict] = []
    for heading, text in sections:
        if not text:
            continue
        if len(text) <= MAX_SECTION_CHARS:
            chunks.append({"section": heading, "text": f"{heading}\n{text}".strip()})
            continue
        # A long section is windowed, with overlap, so a sentence spanning a
        # boundary is not cut in half.
        start = 0
        while start < len(text):
            window = text[start:start + MAX_SECTION_CHARS]
            chunks.append({"section": heading, "text": f"{heading}\n{window}".strip()})
            start += MAX_SECTION_CHARS - WINDOW_OVERLAP
    return chunks


# ── embedding ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], provider: str, model: str, api_key: str) -> list[list[float]]:
    """Embed with the user's own embedding key. Raises so the caller can report it."""
    return build_embeddings(provider, model, api_key).embed_documents(texts)


def embed_query(text: str, provider: str, model: str, api_key: str) -> list[float]:
    """The query side of retrieval, which must use the same model the corpus was
    indexed with — a different one puts the query in an unrelated vector space
    and search returns nonsense without raising anything."""
    return build_embeddings(provider, model, api_key).embed_query(text)


# ── the vector store, with a local fallback ──────────────────────────────────

def pinecone_index():
    """The project's own Pinecone index, or None if the platform has no key.

    One index per project, because an index has a fixed dimension and cannot
    hold vectors from two embedding models. If Pinecone is not configured the
    caller falls back to brute-force cosine over SQLite, which is genuinely fine
    at this corpus size and keeps the project runnable without an account.
    """
    index_name = project_value(SLUG, "pinecone_index")
    if not settings.pinecone_api_key or not index_name:
        return None
    try:
        from pinecone import Pinecone

        return Pinecone(api_key=settings.pinecone_api_key).Index(index_name)
    except Exception as error:  # noqa: BLE001 - degrade rather than fail the request
        logger.warning("Pinecone unavailable, using the local retriever: %s", error)
        return None


def index_workspace(provider: str, embed_model: str, api_key: str,
                    workspace_id: str = DEMO_WORKSPACE) -> dict:
    """Chunk and embed every document that is not already indexed.

    Idempotent: a document whose chunks already exist is skipped. Re-indexing an
    edited document deletes its old chunks first, because appending new ones
    beside the old is how a corpus quietly fills with near-duplicates that
    degrade retrieval without ever raising an error.
    """
    documents = query(
        "SELECT * FROM documents WHERE workspace_id = ? ORDER BY title", (workspace_id,))
    index = pinecone_index()
    conn = database.connect()
    embedded = 0
    skipped = 0

    try:
        for document in documents:
            has_chunks = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?",
                (document["id"],)).fetchone()["n"]
            if has_chunks and document["chunk_count"] == has_chunks:
                skipped += 1
                continue

            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document["id"],))
            pieces = chunk_document(document["body"])
            vectors = embed_texts([piece["text"] for piece in pieces],
                                  provider, embed_model, api_key)

            payload = []
            for ordinal, (piece, vector) in enumerate(zip(pieces, vectors)):
                chunk_id = f"{document['id']}:{ordinal}"
                conn.execute(
                    """INSERT INTO chunks (id, document_id, workspace_id, ordinal, section, text, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (chunk_id, document["id"], workspace_id, ordinal, piece["section"],
                     piece["text"], None if index else json.dumps(vector)))
                if index:
                    payload.append({
                        "id": chunk_id,
                        "values": vector,
                        # Everything here is a filter, never a similarity target.
                        "metadata": {
                            "workspace_id": workspace_id,
                            "document_id": document["id"],
                            "title": document["title"],
                            "doc_type": document["doc_type"],
                            "channel": document["channel"] or "",
                            "segment": document["segment"] or "",
                            "quarter": document["quarter"] or "",
                            "section": piece["section"],
                            "text": piece["text"][:1500],
                        },
                    })

            if payload:
                index.upsert(vectors=payload, namespace=workspace_id)
            conn.execute("UPDATE documents SET chunk_count = ? WHERE id = ?",
                         (len(pieces), document["id"]))
            embedded += len(pieces)
        conn.commit()
    finally:
        conn.close()

    return {"chunks_embedded": embedded, "documents_skipped": skipped,
            "backend": "pinecone" if index else "local"}


# ── query-time filtering ─────────────────────────────────────────────────────

_QUARTER = re.compile(r"\bq([1-4])\s*(?:of\s*)?(20\d\d)|\b(20\d\d)[\s-]*q([1-4])", re.I)
_CHANNELS = {"linkedin": "linkedin", "paid search": "paid_search", "search": "paid_search",
             "email": "email", "nurture": "email", "event": "events", "events": "events"}
_SEGMENTS = {"fintech": "fintech", "insurance": "insurance", "banking": "banking", "bank": "banking"}


def infer_filters(question: str) -> dict:
    """Pull quarter, channel and segment out of the question.

    This is the fix for the failure the project exists to avoid: asking about Q3
    and being handed the Q1 brief. Turning those into filters makes it a lookup
    problem instead of hoping the embedding encoded a date.
    """
    filters: dict[str, str] = {}
    lowered = question.lower()

    match = _QUARTER.search(question)
    if match:
        quarter, year = (match.group(1), match.group(2)) if match.group(1) else (match.group(4), match.group(3))
        filters["quarter"] = f"{year}-Q{quarter}"

    for phrase, value in _CHANNELS.items():
        if phrase in lowered:
            filters["channel"] = value
            break
    for phrase, value in _SEGMENTS.items():
        if phrase in lowered:
            filters["segment"] = value
            break
    return filters


# ── search ───────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    magnitude = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / magnitude if magnitude else 0.0


def _lexical_scores(question: str, rows: list[dict]) -> dict[str, float]:
    """A small BM25-flavoured overlap score.

    Not a full BM25 -- there is no corpus-wide IDF table here -- but it does the
    job the lexical half is for: matching exact tokens like `2024-Q3` that a
    dense embedding blurs.
    """
    terms = {w for w in _WORD.findall(question.lower()) if w not in _STOPWORDS}
    if not terms:
        return {}
    scores: dict[str, float] = {}
    for row in rows:
        words = _WORD.findall(row["text"].lower())
        if not words:
            continue
        counts = sum(words.count(term) for term in terms)
        if counts:
            # Length-normalised, so a long chunk does not win on volume alone.
            scores[row["id"]] = counts / math.sqrt(len(words))
    return scores


def _fuse(dense: list[str], lexical: list[str], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion.

    Combines the two rankings by summing 1/(k + rank). Rank-based on purpose:
    the dense and lexical scores are on entirely different scales and adding
    them directly would mean whichever happens to be larger wins.
    """
    fused: dict[str, float] = {}
    for ranking in (dense, lexical):
        for rank, identifier in enumerate(ranking, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + 1.0 / (k + rank)
    return fused


def search(question: str, provider: str, embed_model: str, api_key: str,
           top_k: int = 5, over_fetch: int = 20,
           workspace_id: str = DEMO_WORKSPACE,
           filters: dict | None = None) -> list[dict]:
    """Hybrid search: filter, then dense + lexical, fused, then cut to top_k."""
    filters = filters or {}
    vector = embed_query(question, provider, embed_model, api_key)
    index = pinecone_index()

    if index:
        condition = {"workspace_id": workspace_id}
        condition.update({key: value for key, value in filters.items() if value})
        response = index.query(vector=vector, top_k=over_fetch, namespace=workspace_id,
                               include_metadata=True, filter=condition or None)
        matches = response.get("matches", []) if isinstance(response, dict) else response.matches
        rows = [{
            "id": match["id"] if isinstance(match, dict) else match.id,
            "score": match["score"] if isinstance(match, dict) else match.score,
            **(match["metadata"] if isinstance(match, dict) else match.metadata),
        } for match in matches]
        # A filter that matches nothing is a worse answer than a slightly wrong
        # one, so an empty filtered result retries unfiltered.
        if not rows and condition != {"workspace_id": workspace_id}:
            return search(question, provider, embed_model, api_key, top_k, over_fetch,
                          workspace_id, filters={})
        dense_ranked = [row["id"] for row in rows]
        lexical_ranked = [identifier for identifier, _ in sorted(
            _lexical_scores(question, rows).items(), key=lambda item: -item[1])]
        fused = _fuse(dense_ranked, lexical_ranked)
        by_id = {row["id"]: row for row in rows}
        ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
        return [{**by_id[identifier], "fused_score": round(score, 5)}
                for identifier, score in ordered if identifier in by_id]

    # ── local retriever ──────────────────────────────────────────────────────
    conditions = ["c.workspace_id = ?"]
    params: list = [workspace_id]
    for column in ("channel", "segment", "quarter"):
        if filters.get(column):
            conditions.append(f"d.{column} = ?")
            params.append(filters[column])

    sql = f"""SELECT c.id, c.text, c.section, c.embedding, d.id AS document_id,
                     d.title, d.doc_type, d.channel, d.segment, d.quarter
              FROM chunks c JOIN documents d ON d.id = c.document_id
              WHERE {' AND '.join(conditions)}"""
    rows = query(sql, tuple(params))
    if not rows and len(conditions) > 1:
        rows = query("""SELECT c.id, c.text, c.section, c.embedding, d.id AS document_id,
                               d.title, d.doc_type, d.channel, d.segment, d.quarter
                        FROM chunks c JOIN documents d ON d.id = c.document_id
                        WHERE c.workspace_id = ?""", (workspace_id,))
    if not rows:
        return []

    scored = []
    for row in rows:
        stored = json.loads(row["embedding"]) if row["embedding"] else None
        row["score"] = _cosine(vector, stored) if stored else 0.0
        scored.append(row)

    dense_ranked = [row["id"] for row in sorted(scored, key=lambda r: -r["score"])[:over_fetch]]
    lexical_ranked = [identifier for identifier, _ in sorted(
        _lexical_scores(question, scored).items(), key=lambda item: -item[1])][:over_fetch]
    fused = _fuse(dense_ranked, lexical_ranked)
    by_id = {row["id"]: row for row in scored}
    ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
    return [{k: v for k, v in by_id[identifier].items() if k != "embedding"}
            | {"fused_score": round(score, 5)}
            for identifier, score in ordered if identifier in by_id]
