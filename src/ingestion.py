"""Persistent cosine indexes with explicit embeddings and atomic document activation."""
import hashlib
import io
import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from filelock import FileLock
from src import config
from src.schemas import ChunkMetadata, DocumentMetadata

INDEX_SCHEMA_VERSION = 2


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def validate_chunking(chunk_size, overlap):
    if type(chunk_size) is not int or type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunking parameters: 0 <= overlap < chunk_size must be true.")


def validate_embedding(vector, dimensions):
    try:
        if isinstance(vector, (str, bytes, dict, set)):
            raise ValueError()
        values = list(vector)
        if len(values) != dimensions or any(isinstance(x, (str, bool, list, tuple)) for x in values):
            raise ValueError()
        values = [float(x) for x in values]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Embedding has invalid dimensions or values") from None
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Vectors must be finite")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must have finite, non-zero norm")
    # Normalize before Chroma's float32 conversion to avoid overflow/underflow.
    return [x / norm for x in values]


def _load_pages(file_path):
    path = Path(file_path)
    if path.suffix.lower() in (".txt", ".md"):
        return [(None, path.read_text(encoding="utf-8"))]
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_raw_text(file_path):
    return "\n".join(text for _, text in _load_pages(file_path))


def _chunk_text_with_metadata(text, document_id, chunk_size=500, overlap=100,
                              file_path=None, *, file_hash="", page_number=None,
                              collection_name="") -> List[ChunkMetadata]:
    validate_chunking(chunk_size, overlap)
    chunks = []
    for start in range(0, len(text), chunk_size - overlap):
        end = min(start + chunk_size, len(text))
        content = text[start:end]
        if not content.strip():
            continue
        # Do not strip: source offsets must reconstruct the exact stored text.
        chunk = ChunkMetadata(
            chunk_id=_hash(json.dumps([document_id, file_hash, collection_name,
                                      page_number, start, end, content])),
            document_id=document_id, file_hash=file_hash,
            file_path=str(file_path or ""), file_name=Path(file_path).name if file_path else "",
            page_number=page_number, char_offset=start, char_length=end - start)
        chunk.set_chunk_text(content)
        chunks.append(chunk)
        if end == len(text):
            break
    return chunks


def _collection_metadata(embedding_model, chunk_size, chunk_overlap):
    validate_chunking(chunk_size, chunk_overlap)
    spec = config.embedding_spec(embedding_model)
    return {"embedding_provider": spec.provider, "embedding_model": spec.model,
            "dimensions": spec.dimensions, "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap, "chunking_algorithm": "page_char_window_v1",
            "index_schema_version": INDEX_SCHEMA_VERSION, "hnsw:space": "cosine"}


def _get_collection_name(embedding_model, chunk_size, chunk_overlap):
    metadata = _collection_metadata(embedding_model, chunk_size, chunk_overlap)
    return "docs_" + _hash(json.dumps(metadata, sort_keys=True))[:48]


def _client():
    if config.CHROMA_CLIENT is not None:
        return config.CHROMA_CLIENT
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR,
                                    settings=Settings(anonymized_telemetry=False))


def get_or_create_collection(embedding_model, chunk_size, chunk_overlap, distance_metric="cosine"):
    if distance_metric != "cosine":
        raise ValueError("Only cosine distance is supported")
    expected = _collection_metadata(embedding_model, chunk_size, chunk_overlap)
    collection = _client().get_or_create_collection(
        name=_get_collection_name(embedding_model, chunk_size, chunk_overlap),
        embedding_function=None, metadata=expected)
    if collection.metadata != expected:
        raise ValueError("Collection metadata mismatch; explicitly rebuild a separate index")
    if (collection.configuration.get("hnsw") or {}).get("space") != "cosine":
        raise ValueError("Collection distance configuration mismatch; rebuild the index")
    return collection


@contextmanager
def _manifest():
    root = Path(config.CHROMA_PERSIST_DIR)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "documents.lock"), timeout=60):
        with _manifest_transaction(root) as db:
            yield db


@contextmanager
def _manifest_transaction(root):
    """Serialize local readers/writers across processes, including staging and cleanup.

    SQLite holds only the active pointer. Chroma writes complete before pointer commit.
    A crash before activation may leave inactive chunks, which retrieval never selects.
    """
    db = sqlite3.connect(root / "documents.sqlite3", timeout=60)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS documents (collection TEXT, identity TEXT, record TEXT NOT NULL, PRIMARY KEY (collection, identity))")
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def _active(db, collection_name, document_id):
    row = db.execute("SELECT record FROM documents WHERE collection=? AND identity=?",
                     (collection_name, document_id)).fetchone()
    return DocumentMetadata.model_validate_json(row[0]) if row else None


def ingest_document(file_path, embedding_model=config.GEMINI_EMBED_MODEL,
                    chunk_size=500, chunk_overlap=100) -> DocumentMetadata:
    validate_chunking(chunk_size, chunk_overlap)
    path = Path(file_path).resolve(strict=True)
    # A consistent byte snapshot is used for both content hash and extraction.
    raw = path.read_bytes()
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    elif path.suffix.lower() in (".txt", ".md"):
        pages = [(None, raw.decode("utf-8"))]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    spec = config.embedding_spec(embedding_model)
    collection = get_or_create_collection(embedding_model, chunk_size, chunk_overlap)
    doc = DocumentMetadata(
        id=_hash(str(path)), file_path=str(path), file_hash=hashlib.sha256(raw).hexdigest(),
        last_modified=path.stat().st_mtime, file_name=path.name, file_size=len(raw),
        embedding_model=spec.model, embedding_provider=spec.provider, dimensions=spec.dimensions,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, collection_name=collection.name,
        page_count=len(pages) if path.suffix.lower() == ".pdf" else None)
    with _manifest() as db:
        previous = _active(db, collection.name, doc.id)
        if previous and previous.file_hash == doc.file_hash:
            return previous
        chunks = [chunk for page, text in pages for chunk in _chunk_text_with_metadata(
            text, doc.id, chunk_size, chunk_overlap, str(path), file_hash=doc.file_hash,
            page_number=page, collection_name=collection.name)]
        if not chunks:
            raise ValueError("Document contains no readable text; previous version remains active")
        # Validate every vector before writing anything. No implicit Chroma embedding function.
        vectors = [validate_embedding(config.get_vector_embedding(c.text, model_name=embedding_model),
                                      spec.dimensions) for c in chunks]
        ids = [c.chunk_id for c in chunks]
        try:
            batch_size = _client().get_max_batch_size()
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                collection.upsert(ids=[c.chunk_id for c in batch], documents=[c.text for c in batch],
                    embeddings=vectors[start:start + batch_size],
                    metadatas=[c.model_dump(exclude={"text"}, exclude_none=True) for c in batch])
            db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)",
                       (collection.name, doc.id, doc.model_dump_json()))
            db.commit()  # Activation after all Chroma batches succeed.
        except Exception:
            # Staging has never been visible through the active pointer.
            try:
                collection.delete(ids=ids)
            except Exception:
                pass  # Orphans remain inactive and can be cleaned on the next successful update.
            raise
        # Keep the lock while removing old/staged versions. Activation is already durable.
        db.execute("BEGIN IMMEDIATE")
        try:
            collection.delete(where={"$and": [{"document_id": doc.id},
                                              {"file_hash": {"$ne": doc.file_hash}}]})
        except Exception:
            pass  # Cleanup failure cannot invalidate the committed active version.
    return doc


def retrieve(query, document_ids, embedding_model, chunk_size, chunk_overlap, top_k=3) -> List[ChunkMetadata]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    identities = [document_ids] if isinstance(document_ids, str) else list(document_ids)
    if not identities:
        return []
    collection = get_or_create_collection(embedding_model, chunk_size, chunk_overlap)
    spec = config.embedding_spec(embedding_model)
    with _manifest() as db:
        active = [_active(db, collection.name, identity) for identity in set(identities)]
        filters = [{"$and": [{"document_id": d.id}, {"file_hash": d.file_hash}]}
                   for d in active if d is not None]
        if not filters:
            return []
        vector = validate_embedding(config.get_vector_embedding(query, model_name=embedding_model), spec.dimensions)
        where = filters[0] if len(filters) == 1 else {"$or": filters}
        result = collection.query(query_embeddings=[vector], where=where,
                                  n_results=top_k, include=["documents", "metadatas", "distances"])
        chunks = []
        for identity, text, metadata in zip(result["ids"][0], result["documents"][0], result["metadatas"][0]):
            chunk = ChunkMetadata.model_validate({**metadata, "text": text})
            if (chunk.chunk_id != identity or _hash(text) != chunk.chunk_text_hash
                    or len(text) != chunk.char_length):
                raise ValueError("Corrupt indexed chunk metadata")
            chunks.append(chunk)
        return chunks


def delete_document_from_collection(document_id, collection):
    with _manifest() as db:
        collection.delete(where={"document_id": document_id})
        db.execute("DELETE FROM documents WHERE collection=? AND identity=?", (collection.name, document_id))
