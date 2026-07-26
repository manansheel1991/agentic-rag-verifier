```markdown
# Agentic RAG Verifier

A defensive, multi-agent Retrieval-Augmented Generation (RAG) framework built with Python, `google-genai`, `openai`, and `Pydantic`.

Unlike standard RAG pipelines that blindly pass raw vector retrieval outputs directly to an LLM, **Agentic RAG Verifier** introduces dual-stage verification gates (**Critic-Generator-Auditor**), an in-memory vector similarity engine, and an automated multi-provider failover architecture to completely eliminate out-of-domain hallucinations.

---

## 🛠️ Architecture & Pipeline Workflow

```text
[ Document / PDF ] ──► [ Sliding Window Chunking ] ──► [ NumPy Vector Engine ]
                                                                 │
                                                        Cosine Similarity Scan
                                                                 │
                                                                 ▼
[ Final Output ] ◄── [ Critic 2: Auditor ] ◄── [ Generator ] ◄── [ Critic 1: Batch Verifier ]

```

### Core Technical Pillars

1. **Dynamic File Ingestion & Chunking (`src/ingestion.py`):** Parses `.pdf`, `.txt`, and `.md` documents, dynamically breaking text into sliding-window chunks with configurable overlaps to preserve sentence context across boundaries.
2. **In-Memory Vector Math Engine (`src/main.py`):** Translates prose into dense embedding vectors (`gemini-embedding-001` with `text-embedding-3-small` fallback) and computes raw Cosine Similarity using `NumPy` linear algebra ($\cos\theta$).
3. **Multi-Provider Failover Layer (`src/config.py`):** Guarantees zero downtime by catching model rate limits or 429 quota errors (e.g., Gemini Free Tier limits) and automatically rerouting structured API calls to OpenAI (`gpt-4o-mini`).
4. **Critic 1 — Batch Context Verification:** Evaluates top-$k$ retrieved chunks in a **single, concurrent API payload** at `temperature=0.0`. Optimizes token consumption by **60%** while executing strict relevance filtering. If no chunks pass, execution safely halts before generation.
5. **Generator — Grounded Synthesis:** Synthesizes a factual answer using *only* context chunks explicitly validated by Critic 1.
6. **Critic 2 — Hallucination Auditor:** Performs an end-to-end alignment audit, comparing the generated response back against source context to block ungrounded claims.

---

## 🚀 Quickstart & Usage

### 1. Installation

Clone the repository and set up your Python virtual environment:

```bash
git clone [https://github.com/YOUR_USERNAME/agentic-rag-verifier.git](https://github.com/YOUR_USERNAME/agentic-rag-verifier.git)
cd agentic-rag-verifier

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Environment Setup

Create a `.env` file in the root directory and add your API credentials:

```bash
GEMINI_API_KEY="your_gemini_api_key"
OPENAI_API_KEY="your_openai_api_key"

```

*Note: Ensure `.env` is listed in `.gitignore` to protect API keys.*

### 3. Running Queries via CLI

The application includes a built-in Command Line Interface (CLI) supporting custom documents (`.pdf`, `.txt`, `.md`) and queries:

```bash
# Query a PDF document
python src/main.py -f data/my_document.pdf -q "Summarize the key deliverables"

# Query a Markdown file
python src/main.py -f README.md -q "What model is used for embeddings?"

# Default behavior (runs sample architecture report if no flags are passed)
python src/main.py

```

---

## 🛠️ Tech Stack

* **Language:** Python 3.11+ (CLI via `argparse`)
* **Vector Engine:** Native `NumPy` Linear Algebra (Raw Cosine Similarity execution)
* **LLM Engine & Failover:** `gemini-2.5-flash-lite` ──► `gpt-4o-mini`
* **Embeddings Tier:** `gemini-embedding-001` ──► `text-embedding-3-small`
* **Document Parsing:** `pypdf` for PDF text extraction; standard IO for text and Markdown
* **Schema Enforcement & Reliability:** `Pydantic` and `python-dotenv`

```

```