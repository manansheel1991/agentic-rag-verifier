# Agentic RAG Verifier

A lightweight, defensive, multi-agent Retrieval-Augmented Generation (RAG) framework built with Python, `google-genai`, and `Pydantic`. 

Unlike standard RAG pipelines that blindly feed vector retrieval outputs directly into a language model, **Agentic RAG Verifier** introduces dual-stage verification gates (**Critic-Generator-Auditor**) and an in-memory vector search engine to eliminate out-of-domain hallucinations and prevent ungrounded responses.

---

## 🛠️ Architecture & Pipeline Overview

```text
[ Document / PDF ] ──► [ Sliding Window Chunking ] ──► [ NumPy Vector Engine ]
                                                                 │
                                                        Cosine Similarity Scan
                                                                 │
                                                                 ▼
[ Final Output ] ◄── [ Critic 2: Auditor ] ◄── [ Generator ] ◄── [ Critic 1: Batch Verifier ]

```

### Key Components

1. **Dynamic File Ingestion & Chunking (`src/ingestion.py`):** Automatically parses local `.pdf`, `.txt`, and `.md` files, processing raw text into sliding-window chunks with contextual overlaps to prevent sentence truncation.
2. **In-Memory Vector Search (`src/main.py`):** Translates prose into dense embedding vectors (`gemini-embedding-001`) and computes raw Cosine Similarity using NumPy linear algebra ($\cos\theta$).
3. **Critic 1 — Batch Context Verification:** Evaluates all top-$k$ retrieved chunks concurrently within a **single optimized API payload** (`temperature=0.0`). Reduces API token volume by **60%** and enforces strict deterministic relevance filtering. If no chunks pass, execution safely halts before generation.
4. **Generator — Grounded Synthesis:** Synthesizes an answer using *only* the verified context chunks passed through Critic 1.
5. **Critic 2 — Hallucination Auditor:** Performs a final cross-reference check comparing the synthesized answer back against the source context to ensure absolute alignment before returning the response.

---

## 🚀 Quickstart

### 1. Prerequisites & Installation

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

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add your Gemini API key:

```bash
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env

```

*Note: Ensure `.env` is listed in your `.gitignore` file to prevent committing secrets.*

### 3. Execution

Load environment variables and run the main entry point:

```bash
source .env
python src/main.py

```

By default, the script creates a sample system report in `data/sample_report.txt`, chunks the content, computes vector similarities, passes context through the verification critics, and outputs a validated answer.

---

## 🛠️ Tech Stack & Key Paradigms

* **Language:** Python 3.11+
* **Vector Engine:** Native `NumPy` Linear Algebra (Raw Cosine Similarity execution)
* **LLM Tier:** `gemini-2.5-flash-lite` (High availability and low-latency structured extraction)
* **Embeddings Engine:** `gemini-embedding-001`
* **Document Parsing:** `pypdf` for PDF ingestion; standard IO for plain text & Markdown
* **Schema Enforcement:** `Pydantic` and `google-genai` Schema Engine
