# Agentic RAG Verifier

A small Python/Pydantic RAG pipeline with persistent local Chroma storage:

```text
Document → page-aware chunks → explicit embeddings → persistent cosine index
Query → same embedding model → selected documents → relevance critic
      → answer generator → grounding auditor → typed PipelineResult
```

The critics reduce unsupported answers; model-based verification does not guarantee factual correctness. No agent framework is used.

## Install and configure

Use Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` and/or `OPENAI_API_KEY` in `.env`. Do not commit credentials. The default embedding model requires Gemini; an OpenAI generation fallback does **not** replace the embedding provider.

- `CHROMA_PERSIST_DIR`: local index directory (default `./chroma_db`). Keep the Chroma files, `documents.sqlite3`, and `documents.lock` together. When choosing a custom directory, use a path outside the repository or add that directory to `.gitignore`.
- `API_TIMEOUT_SECONDS`: per-request timeout (default 30 seconds). SDK retries are disabled; the application makes at most three attempts for rate limits, timeouts, connection failures and transient server failures, with bounded backoff. Authentication, invalid-request and malformed-output failures are not retried on that provider.

Clients initialize lazily. Importing the application does not call a model or create an index. Actual CLI queries use the configured APIs and may incur provider charges.

## CLI and Python usage

Both invocation forms are supported:

```bash
python src/main.py -f data/my_document.pdf -q "Summarize the deliverables"
python -m src.main -f README.md -q "How is ingestion made reliable?"
```

Select OpenAI embeddings explicitly to build a separate index:

```bash
python -m src.main -f data/report.md -q "What changed?" \
  --embedding_model openai/text-embedding-3-small \
  --generation_model openai/gpt-4o-mini \
  --chunk_size 500 --chunk_overlap 100
```

Existing flags remain available; no arguments runs the two sample-report queries. Text, Markdown and text-based PDFs are supported. Scanned PDFs require external OCR.

```python
from src.main import run_pipeline

result = run_pipeline("data/report.md", "What changed?")
print(result.model_dump_json(indent=2))
```

Results include outcome, verified answer, typed sources (including text), critic audit, stage timings, actual model identities, and token totals when reported by providers. Failures retain completed timings and diagnostics. An unfaithful or unaudited answer is withheld. `generation_model_used` is the model that synthesized the answer; `model_identities` includes all responding models across the critics and generator. Token totals include available usage; providers do not always return usage, and missing usage is represented by `None`, not zero.

For multiple selected documents, ingest each with the same configuration and call `src.ingestion.retrieve(query, [doc1.id, doc2.id], model, chunk_size, overlap)`. Only those documents' active versions are searched.

## Index guarantees

- Chroma uses local `PersistentClient`, explicit embeddings, `embedding_function=None`, and cosine distance. No default embedding model is downloaded or invoked.
- Collection names hash provider, model, dimensions, chunk size/overlap, chunking algorithm and schema version. Metadata is strictly checked on every open; mismatches fail instead of being silently reused.
- Supported embedding identities are `gemini/gemini-embedding-001` (3072 dimensions), `openai/text-embedding-3-small` (1536) and `openai/text-embedding-3-large` (3072). Bare model names are also accepted. Additional models require explicitly registering their provider/model/dimensions and rebuilding a separate index. Equal dimensions never establish model compatibility.
- Document identity hashes the canonical absolute source path. Content versions hash the exact source bytes. Renaming/moving a file creates a new identity; selecting the new identity excludes the old one.
- Chunk IDs deterministically include document identity, content hash, collection, page, offsets and text. Stored metadata retains name, path, content hash, page number (PDFs), and character offsets within the extracted page/text. Whitespace is preserved so slicing by offsets reproduces the stored text.
- Nonempty short chunks are retained. `0 <= overlap < chunk_size` is required even for empty text. Empty, unreadable and unsupported files fail clearly; finite, nonzero, correctly dimensioned vectors are required for ingestion and queries.
- Unchanged content skips embeddings even if modification time changes. All embeddings are validated and all Chroma batches staged before committing the active document pointer to a small SQLite manifest. Failed updates preserve the previous active version.
- A local file lock serializes manifest readers and writers through staging, activation and cleanup. Successful updates delete old chunks after activation. Crash leftovers or failed cleanup may leave **inactive** chunks on disk; retrieval always filters by the manifest's selected document and active content hash. A later successful changed ingestion removes old versions.
- Queries use exactly the selected collection's provider/model. Provider unavailability is an error; switching embedding models requires explicitly ingesting into a different collection. Text generation can independently fall back from Gemini to OpenAI.
- Critic evaluations must cover the exact supplied chunk-ID set once each. Unknown, duplicate, missing or malformed evaluations fail closed.

Schema v2 intentionally creates new collections and does not migrate earlier experimental indexes whose model provenance is unreliable. Re-ingest source files explicitly; do not relabel an old collection's metadata.

## Offline tests

```bash
python -m pytest -q
```

Tests use temporary local Chroma directories, mocked model providers, blocked Python network connections and placeholder keys. They cover fresh-process persistence, idempotency, changed/failed updates, failed cleanup, document isolation, real PDF pages and offsets, metadata/model mismatch, invalid vectors and chunking, strict critics, provider failures, timeouts/retries, generation fallback, usage, typed results and CLI behavior. No paid API calls or embedding downloads are needed.

The test suite verifies provider integration through mocks; live credentials, live model compatibility and provider quality are not tested. Local readers/writers are serialized for correctness, so this implementation is intended for modest local workloads rather than distributed serving. Text extraction supplies page-relative character offsets, not PDF bounding boxes. Model critics can still make mistakes.
