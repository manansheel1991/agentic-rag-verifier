# Agentic RAG with "Verify-before-Synthesize" Architecture

A defensive, self-correcting Retrieval-Augmented Generation (RAG) pipeline built with Python and the native `google-genai` SDK. It introduces multi-stage verification checkpoints using structured LLM outputs to mitigate retrieval noise and completely eliminate hallucinated generations.

---

## 🔍 The Problem & The Architecture

Standard (naive) RAG pipelines blindly trust the output of vector similarity math. If a semantic database search retrieves noisy, incomplete, or out-of-domain text fragments, the generator LLM constructs a confident, polished hallucination.

This project implements an **Agentic Loop** that introduces quality gates across the execution lifespan:

              [ User Query ]
                    │
                    ▼
        [ Dense Vector Retrieval ]
                    │
                    ▼
           [ Retrieved Context ]
                    │
                    ▼
        ┌───► [ Critic Agent ] ◄──────────────┐
        │     (Is this context relevant?)     │
        │           │                         │
  No (Low Score)    │ Yes (High Score)        │
        │           ▼                         │
        │     [ Generator LLM ]               │
        │           │                         │
        │           ▼                         │
        │     [ Response Critic ]             │
        │     (Is it hallucinated?)           │
        │           │                         │
        │           ├─── No (Faithful) ──► [ Final Output ]
        │           │
        └───────────┴─── Yes (Hallucinated: Block Execution)

### Key Safety Gates
1. **Critic 1 (Context Verification):** Evaluates incoming fragments at `temperature=0.0`. Drops irrelevant text pieces. If zero pieces pass, a **Defensive Fallback Mechanism** stops the pipeline from generating an answer entirely.
2. **Critic 2 (Response Auditor):** Evaluates the final natural language answer back against the source fragments to enforce cross-referencing alignment before surfacing data to the user.

---

## 🛠️ Tech Stack & Key Paradigms

* **Language:** Python 3.11+
* **LLM Engine:** `gemini-2.5-flash`
* **Orchestration:** Pure Python control flow (Eliminates framework abstraction bloat)
* **Structured Data Guardrails:** `Pydantic` and `google-genai` Schema Engine

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/agentic-rag-verifier.git](https://github.com/YOUR_USERNAME/agentic-rag-verifier.git)
cd agentic-rag-verifier
pip install -r requirements.txt
