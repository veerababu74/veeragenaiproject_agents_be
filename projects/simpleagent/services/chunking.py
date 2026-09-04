"""Text extraction and the four chunking strategies.

Extraction is structural rather than flat: PDFs, DOCX files and CSVs are read
into typed blocks (heading, paragraph, bullet, table, image) and only then
flattened. Three of the four strategies work on the flattened text, but
context-aware needs to know what each piece of text *was*, and recovering that
from a flat string is guesswork.
"""

import csv
import io
import logging
import re

logger = logging.getLogger("simpleagent.chunking")

STRATEGIES = ("fixed", "recursive", "semantic", "context_aware")
DEFAULT_STRATEGY = "recursive"

SUPPORTED_TYPES = {"pdf", "txt", "docx", "csv"}

# Semantic chunking splits where consecutive sentences stop being about the same
# thing. 0.72 is low enough to keep a paragraph together and high enough to cut
# when a document changes subject.
SEMANTIC_SIMILARITY_THRESHOLD = 0.72
SEMANTIC_MAX_SENTENCES = 400

_BULLET = re.compile(r"^\s*(?:[-*•▪◦]|\(?\d{1,2}[.)])\s+")
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+\S|[A-Z0-9][A-Z0-9 \-_/&]{2,79})\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|\n{2,}")


class ExtractionError(Exception):
    pass


# ── extraction ──────────────────────────────────────────────────────────────

def _block(kind, text):
    return {"kind": kind, "text": text.strip()}


def _classify_line(line: str) -> str:
    if _BULLET.match(line):
        return "bullet"
    if _HEADING.match(line) and len(line.strip()) <= 80:
        return "heading"
    return "paragraph"


def _blocks_from_plain_text(text: str) -> list[dict]:
    blocks = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        kind = _classify_line(line)
        # Consecutive paragraph lines belong to the same paragraph; a hard wrap
        # in the source is not a semantic boundary.
        if kind == "paragraph" and blocks and blocks[-1]["kind"] == "paragraph":
            blocks[-1]["text"] += " " + line
        else:
            blocks.append(_block(kind, line))
    return blocks


def _extract_pdf(content: bytes) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    blocks: list[dict] = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            blocks.extend(_blocks_from_plain_text(text))
        # A page whose text is empty but which carries XObjects is a scan or a
        # figure. Recording it keeps the position of an image in the document
        # visible instead of silently dropping the page.
        elif "/XObject" in (page.get("/Resources") or {}):
            blocks.append(_block("image", f"[image or scanned content on page {number}]"))
    return blocks


def _extract_docx(content: bytes) -> list[dict]:
    from docx import Document

    document = Document(io.BytesIO(content))
    blocks: list[dict] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if style.startswith("heading") or style == "title":
            blocks.append(_block("heading", text))
        elif "list" in style or _BULLET.match(text):
            blocks.append(_block("bullet", text))
        else:
            blocks.append(_block("paragraph", text))
    for table in document.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        rows = [row for row in rows if row.replace("|", "").strip()]
        if rows:
            blocks.append(_block("table", "\n".join(rows)))
    if any(part.content_type.startswith("image/") for part in document.part.package.parts):
        blocks.append(_block("image", "[document contains embedded images]"))
    return blocks


def _extract_csv(content: bytes) -> list[dict]:
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []
    header = " | ".join(cell.strip() for cell in rows[0])
    blocks = [_block("heading", header)]
    # Rows are grouped rather than emitted one by one: a single spreadsheet row
    # is far too small to embed usefully on its own.
    group: list[str] = []
    for row in rows[1:]:
        group.append(" | ".join(cell.strip() for cell in row))
        if len(group) == 25:
            blocks.append(_block("table", header + "\n" + "\n".join(group)))
            group = []
    if group:
        blocks.append(_block("table", header + "\n" + "\n".join(group)))
    return blocks


def extract_blocks(content: bytes, file_type: str) -> list[dict]:
    try:
        if file_type == "txt":
            return _blocks_from_plain_text(content.decode("utf-8", errors="replace"))
        if file_type == "pdf":
            return _extract_pdf(content)
        if file_type == "docx":
            return _extract_docx(content)
        if file_type == "csv":
            return _extract_csv(content)
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError(f"Could not read the {file_type.upper()} file: {error}") from error
    raise ExtractionError(f"Unsupported file type: {file_type}")


def blocks_to_text(blocks: list[dict]) -> str:
    return "\n\n".join(block["text"] for block in blocks if block["text"])


# ── strategies ──────────────────────────────────────────────────────────────

def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[start:start + size].strip() for start in range(0, len(text), step) if text[start:start + size].strip()]


def chunk_recursive(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _cosine(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def chunk_semantic(text: str, size: int, api_key: str, embedding_model: str) -> list[str]:
    """Group consecutive sentences while they stay on the same topic.

    Falls back to recursive splitting when there is nothing to compare (a single
    sentence) — an embedding call per upload is not worth making to discover
    there is only one boundary candidate.
    """
    from core.embeddings import embed_texts

    sentences = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    if len(sentences) < 2:
        return chunk_recursive(text, size, size // 8)
    sentences = sentences[:SEMANTIC_MAX_SENTENCES]

    vectors = embed_texts(api_key, embedding_model, sentences, "RETRIEVAL_DOCUMENT")
    chunks: list[str] = []
    current = [sentences[0]]
    for index in range(1, len(sentences)):
        similarity = _cosine(vectors[index - 1], vectors[index])
        pending = len(" ".join(current)) + len(sentences[index])
        if similarity < SEMANTIC_SIMILARITY_THRESHOLD or pending > size:
            chunks.append(" ".join(current))
            current = [sentences[index]]
        else:
            current.append(sentences[index])
    if current:
        chunks.append(" ".join(current))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def chunk_context_aware(blocks: list[dict], size: int, overlap: int) -> list[str]:
    """Pack structural blocks into chunks without breaking their structure.

    A heading starts a new chunk and is repeated at the top of every chunk that
    continues under it, so a retrieved fragment still says what section it came
    from. A table is never split across chunks unless it is larger than the
    chunk size on its own, and bullets stay with the bullets around them.
    """
    chunks: list[str] = []
    heading = ""
    current: list[str] = []
    current_length = 0

    def flush():
        nonlocal current, current_length
        if current:
            body = "\n".join(current)
            chunks.append(f"{heading}\n{body}".strip() if heading else body)
            current = []
            current_length = 0

    for block in blocks:
        text = block["text"]
        if not text:
            continue
        if block["kind"] == "heading":
            flush()
            heading = text
            continue
        if block["kind"] == "table":
            flush()
            if len(text) <= size:
                chunks.append(f"{heading}\n{text}".strip() if heading else text)
            else:
                # An oversized table still has to be cut, but on row boundaries.
                rows = text.split("\n")
                buffer: list[str] = []
                for row in rows:
                    if sum(len(r) + 1 for r in buffer) + len(row) > size and buffer:
                        chunks.append(f"{heading}\n" + "\n".join(buffer) if heading else "\n".join(buffer))
                        buffer = []
                    buffer.append(row)
                if buffer:
                    chunks.append(f"{heading}\n" + "\n".join(buffer) if heading else "\n".join(buffer))
            continue
        if current_length + len(text) > size and current:
            tail = current[-1] if overlap and len(current[-1]) <= overlap else ""
            flush()
            if tail:
                current, current_length = [tail], len(tail)
        current.append(text)
        current_length += len(text) + 1

    flush()
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def chunk_document(blocks, strategy, size, overlap, embedding_api_key="", embedding_model=""):
    """Apply one strategy and report what it did, for the upload summary."""
    text = blocks_to_text(blocks)
    if not text.strip():
        raise ExtractionError("No readable text was found in this document")

    if strategy == "fixed":
        chunks = chunk_fixed(text, size, overlap)
    elif strategy == "semantic":
        chunks = chunk_semantic(text, size, embedding_api_key, embedding_model)
    elif strategy == "context_aware":
        chunks = chunk_context_aware(blocks, size, overlap)
    else:
        chunks = chunk_recursive(text, size, overlap)

    if not chunks:
        raise ExtractionError("The document produced no usable chunks")
    return chunks
