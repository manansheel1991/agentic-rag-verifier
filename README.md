# Personal RAG Verifier

A small Python/Pydantic project for answering questions over selected documents and
checking individual claims against cited passages. It includes persistent local Chroma,
a CLI, a Streamlit demo, and a human-reviewed development evaluation harness. This is
a **verification pipeline**: no agent framework or answer-recovery loop is implemented.

```text
Documents → page-aware chunks → explicit embeddings → persistent Chroma
Question → same embedding model → selected-document retrieval
                                  ├─ Basic: structured claims → display (unverified)
                                  └─ Verifier: relevance gate → structured claims
                                               → claim audit → citation gates → display
```

## Install

Use Python **3.11+**, with `python3` pointing to that interpreter. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Offline mode needs no credentials. For live queries, run `cp .env.example .env` and
set `GEMINI_API_KEY` or `OPENAI_API_KEY` locally. `.env` is ignored by Git; keys are
never entered into the UI. `CHROMA_PERSIST_DIR` defaults to `./chroma_db` and
`API_TIMEOUT_SECONDS` to 30 seconds. Live API queries can incur charges.

## Streamlit demo

```bash
# Default: clearly labelled offline mock, no credentials or paid API calls
python -m streamlit run app.py

# Explicit live mode: submitting Ask can call paid providers
python -m streamlit run app.py -- --live
```

Open the printed localhost URL, select sample documents, and choose **Verifier** or
**Basic RAG**. Both use the same document selector, retrieval settings, generator and
answer/citation format. Live mode offers Gemini or OpenAI model pairs and disables
fallback so the selected model remains explicit. There are no uploads or deployment settings.

With the default documents selected:

1. Click **Supported example**, then **Ask**. Harbor's peak rate is **420 events per
   second**. In offline Verifier mode the claim is labelled `supported`; expand the
   passage from `harbor-operations.md` to inspect its text and stable ID. Basic mode
   labels the same claim `not_verified`.
2. Click **Abstention example**, then **Ask**. Harbor's annual revenue is absent from
   the corpus. The offline demo shows **Abstained** and the unanswered question.

These offline presets are **scripted fixtures, not model performance**. Other offline
questions generally abstain; live models can behave differently. Real local Chroma
and existing pipeline gates still run. Changing inputs hides stale results until you
ask again. The index loads/updates only when you submit a question.

The UI distinguishes **answered**, **partial**, **abstained**, and **error**. Only
supported claims enter the verifier's answer; withheld claims are audit diagnostics
inside an expander. Passages retain document names and PDF pages where available
(the included Markdown files have no page numbers). Unanswered parts and source
disagreements remain visible. Basic claims are not factually verified.

Query latency, model calls and available token usage exclude indexing, shown separately.
Missing usage is **unavailable**, not zero; known totals can be partial. Mock timings
are local fixture timings, not real model speed/cost. Streamlit telemetry is disabled;
the demo does not render credentials, SDK request objects or raw exceptions.

## CLI and Python

```bash
# Live query, using Gemini by default
python -m src.main -f evaluation/corpus/harbor-operations.md \
  -q "What is Harbor's peak processing rate?"

# Existing script entry point with explicit OpenAI models
python src/main.py -f evaluation/corpus/harbor-operations.md \
  -q "How long are event logs retained?" \
  --embedding_model openai/text-embedding-3-small \
  --generation_model openai/gpt-4o-mini --chunk_size 500 --chunk_overlap 100
```

No arguments runs the existing two sample queries **with live providers**. Text,
Markdown and text-based PDFs are supported. `src.main.run_pipeline(file_path, question)`
returns a typed result with claims, sources, audits, unanswered parts, timings, models
and available usage. Operational failures retain error fields in an `abstained` result;
the demo presents these as **error**.

## Persistence and verification

Collections pin embedding provider/model, dimensions, chunking configuration and schema
version, with explicit cosine distance and no Chroma default embedding generation.
Queries use exactly that embedding identity; equal dimensions do not establish model
compatibility. Metadata mismatches fail, and embedding providers never silently switch.

Unchanged documents skip embedding. Updates stage and validate new chunks before
activating the content version in a SQLite manifest; failed updates retain the previous
working version. Retrieval filters selected document identities and active content hashes.
Chunks preserve names, stable IDs, hashes, page references and source character offsets.
Chunking requires `0 <= overlap < chunk_size`; vectors must be finite, nonzero and
correctly dimensioned.

The live demo reuses `CHROMA_PERSIST_DIR` and matching existing collections. Offline
mock storage uses its separate `demo_mock/` subdirectory and mock embedding identity.
Keep Chroma, `documents.sqlite3` and lock files together. Generated indexes are ignored;
ignore custom directories too. Earlier experimental indexes need explicit re-ingestion.

All model stages treat documents as untrusted data. The auditor judges each claim
`supported`, `contradicted` or `insufficient_evidence`. Separate Python gates validate
citation IDs, evaluation coverage and exact excerpt containment. These prove provenance,
**not semantic truth**. Dropped claims keep their question parts unresolved. Bounded
transient API retries are transport retries, not answer recovery.

## Evaluation

The [evaluation guide](evaluation/README.md) covers the six-document, 25-question
synthetic development set, its CC0 license, human labels, metric denominators and costs.
It requires review and is **not an independent benchmark**. Only basic RAG and the
existing verifier are compared.

```bash
# Offline harness check; use a new output directory for each run
python -m src.evaluation run --mode mock --output eval_runs/mock-check

# Explicit live API evaluation; can incur charges
python -m src.evaluation run --mode live --output eval_runs/live-dev \
  --embedding-model openai/text-embedding-3-small \
  --generation-model openai/gpt-4o-mini

# After filling human review labels in review.csv
python -m src.evaluation report \
  --results eval_runs/live-dev/results.json \
  --review eval_runs/live-dev/review.csv
```

Paired JSON and CSV outputs retain model/configuration identities and corpus/code hashes.
Human-dependent metrics remain unscored until review; the system auditor is not the
judge. Uncited claims are evaluated too: missing citations alone do not establish
incorrectness. **No reviewed live-run results are available in this checkout**, so no
measured accuracy or performance improvements are reported.

## Offline tests

```bash
python -m pytest -q
```

Tests use mocked providers, blocked model network calls and temporary Chroma storage.
They cover indexing/retrieval reliability, claim gates, provider failures, evaluation
bookkeeping and Streamlit widgets/presets. Mocks validate enforcement, not model judgment
accuracy. GitHub Actions installs dependencies and runs this suite without provider secrets.

## Limitations

Verification can accept unsupported claims, miss conflicts/omissions, or reject correct
answers. The demo retrieves only the top three chunks. Reliability adds latency and
model calls; no unmeasured quality or speed gain is claimed. There is no regeneration
or recovery. Scanned PDFs need OCR; PDF offsets are extracted-text positions, not page
bounding boxes. Local index access and demo requests are serialized for modest workloads.
Live provider quality/compatibility and hosted CI execution require separate verification.
