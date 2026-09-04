"""Uploading the documents the retriever tool searches.

The upload is synchronous on purpose. Extraction, chunking, embedding and
indexing all have to succeed for a document to be searchable, and doing them
behind a job queue would only move the failure somewhere the user cannot see it.
If indexing fails after the file is stored, the stored copy is removed again so
a half-ingested document never lingers.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from core.auth import current_user_id
from core.embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, EmbeddingError, embed_texts
from core.storage import BucketError
from core.vectors import VectorStoreError
from projects.simpleagent.database import get_db
from projects.simpleagent.resources import bucket, vectors
from projects.simpleagent.services.chunking import (DEFAULT_STRATEGY, STRATEGIES, SUPPORTED_TYPES,
                                                    ExtractionError, chunk_document, extract_blocks)
from projects.simpleagent.services.llm_provider import save_key

logger = logging.getLogger("simpleagent.documents")

router = APIRouter(prefix="/documents", tags=["documents"])

QUOTA_BYTES = 5 * 1024 * 1024

STRATEGY_LABELS = {
    "fixed": "Fixed size — a plain character window with overlap",
    "recursive": "Recursive — splits on paragraph, then sentence, then word boundaries",
    "semantic": "Semantic — starts a new chunk where the meaning shifts (uses your Gemini key)",
    "context_aware": "Context-aware — respects headings, bullet lists and tables",
}


def _used_bytes(user_id):
    conn = get_db()
    total = conn.execute("SELECT COALESCE(SUM(file_size), 0) AS used FROM documents WHERE user_id=?",
                         (user_id,)).fetchone()["used"]
    conn.close()
    return total or 0


def _document(document_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


@router.get("/options")
async def options():
    return {
        "strategies": [{"id": key, "label": STRATEGY_LABELS[key], "needs_embedding_key": key == "semantic"}
                       for key in STRATEGIES],
        "default_strategy": DEFAULT_STRATEGY,
        "embedding_models": list(EMBEDDING_MODELS),
        "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
        "file_types": sorted(SUPPORTED_TYPES),
        "quota_bytes": QUOTA_BYTES,
    }


@router.get("")
async def list_documents(user_id: str = Depends(current_user_id)):
    conn = get_db()
    rows = conn.execute("SELECT * FROM documents WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
    conn.close()
    used = _used_bytes(user_id)
    return {
        "documents": [dict(row) for row in rows],
        "quota_bytes": QUOTA_BYTES,
        "used_bytes": used,
        "remaining_bytes": max(0, QUOTA_BYTES - used),
    }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form(DEFAULT_STRATEGY),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
    embedding_api_key: str = Form(...),
    user_id: str = Depends(current_user_id),
):
    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if extension not in SUPPORTED_TYPES:
        raise HTTPException(400, f"Only {', '.join(sorted(SUPPORTED_TYPES)).upper()} files are supported")
    if chunk_strategy not in STRATEGIES:
        raise HTTPException(400, f"Unknown chunking strategy: {chunk_strategy}")
    if embedding_model not in EMBEDDING_MODELS:
        raise HTTPException(400, f"Choose one of these embedding models: {', '.join(EMBEDDING_MODELS)}")
    embedding_api_key = embedding_api_key.strip()
    if not embedding_api_key:
        raise HTTPException(400, "A Google Gemini API key is required to create embeddings")

    chunk_size = max(200, min(chunk_size, 4000))
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    content = await file.read()
    used = _used_bytes(user_id)
    if used + len(content) > QUOTA_BYTES:
        remaining = max(0, QUOTA_BYTES - used) / 1024 / 1024
        raise HTTPException(
            400, f"That would exceed your {QUOTA_BYTES // 1024 // 1024} MB allowance — {remaining:.1f} MB left. "
                 "Delete a document to free space.")
    if not content:
        raise HTTPException(400, "The file is empty")

    # The key is saved so the retriever tool can embed queries later with the
    # same model the documents were built with.
    save_key(user_id, "google", embedding_api_key)

    document_id = str(uuid.uuid4())
    remote_path = bucket.path_for(user_id, document_id, extension)
    now = datetime.utcnow().isoformat()

    conn = get_db()
    conn.execute(
        """INSERT INTO documents (id, user_id, file_name, file_type, file_size, remote_path, embedding_model,
                                  chunk_strategy, chunk_size, chunk_overlap, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,'processing',?)""",
        (document_id, user_id, file.filename, extension, len(content), remote_path, embedding_model,
         chunk_strategy, chunk_size, chunk_overlap, now))
    conn.commit()
    conn.close()

    structure: dict[str, int] = {}
    try:
        blocks = extract_blocks(content, extension)
        for block in blocks:
            structure[block["kind"]] = structure.get(block["kind"], 0) + 1
        chunks = chunk_document(blocks, chunk_strategy, chunk_size, chunk_overlap,
                                embedding_api_key, embedding_model)
        embeddings = embed_texts(embedding_api_key, embedding_model, chunks, "RETRIEVAL_DOCUMENT")
        bucket.upload(content, remote_path)
        try:
            vectors.upsert(user_id, document_id, file.filename, chunks, embeddings)
        except VectorStoreError:
            bucket.delete(remote_path)
            raise
        conn = get_db()
        conn.execute("UPDATE documents SET status='ready', chunk_count=? WHERE id=?", (len(chunks), document_id))
        conn.commit()
        conn.close()
        preview = [chunk[:400] for chunk in chunks[:3]]
    except (ExtractionError, EmbeddingError, BucketError,
            VectorStoreError, ValueError) as error:
        conn = get_db()
        conn.execute("UPDATE documents SET status='error', error_message=? WHERE id=?", (str(error), document_id))
        conn.commit()
        conn.close()
        preview = []
    except Exception as error:
        logger.exception("Unexpected failure while ingesting %s", file.filename)
        conn = get_db()
        conn.execute("UPDATE documents SET status='error', error_message=? WHERE id=?",
                     ("The document could not be processed", document_id))
        conn.commit()
        conn.close()
        preview = []

    return {**_document(document_id), "chunk_preview": preview, "structure": structure}


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, user_id: str = Depends(current_user_id)):
    conn = get_db()
    row = conn.execute("SELECT * FROM documents WHERE id=? AND user_id=?", (document_id, user_id)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Document not found")
    document = dict(row)
    conn.close()

    if document["status"] == "ready":
        try:
            vectors.delete_document(user_id, document_id, document["chunk_count"] or 0)
        except VectorStoreError:
            logger.warning("Could not remove vectors for %s", document_id)
        if document["remote_path"]:
            try:
                bucket.delete(document["remote_path"])
            except BucketError:
                logger.warning("Could not remove the stored file for %s", document_id)

    conn = get_db()
    conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    conn.commit()
    conn.close()
