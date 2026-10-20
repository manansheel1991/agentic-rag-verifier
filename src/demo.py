"""Small UI adapter over existing indexing, retrieval and paired pipeline runners."""
import json
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from src import config
from src.evaluation import measure, run_arm, usage
from src.evaluation_mock import MOCK_EMBEDDING, mock_providers
from src.ingestion import ingest_document, retrieve

CORPUS = Path(__file__).resolve().parents[1] / "evaluation" / "corpus"
SUPPORTED_QUESTION = "What is Harbor's peak processing rate?"
ABSTENTION_QUESTION = "What is Harbor's annual revenue?"
PRESET_FACT = "The Harbor service processes 420 events per second at peak."
LIVE_MODELS = {
    "Gemini": ("gemini/" + config.GEMINI_EMBED_MODEL, "gemini/" + config.GEMINI_GEN_MODEL),
    "OpenAI": ("openai/" + config.OPENAI_EMBED_MODEL, "openai/" + config.OPENAI_GEN_MODEL),
}
# The local mock context patches SDK clients; serialize demo sessions while it is active.
_DEMO_LOCK = threading.RLock()


def sample_documents():
    return {path.name: path for path in sorted(CORPUS.glob("*.md"))}


def _mock_generation(model, messages, response_format, **kwargs):
    """Two scripted examples, not a semantic model or evaluation judge."""
    prompt = messages[0]["content"]
    schema = response_format["json_schema"]["name"]
    query = prompt.split("Query: ", 1)[1].split("\n", 1)[0].strip()
    supported = query.casefold() == SUPPORTED_QUESTION.casefold()
    if schema == "BatchContextVerification":
        chunks = json.JSONDecoder().raw_decode(prompt.split("Available Document Chunks:\n", 1)[1])[0]
        data = {"evaluations": [{"chunk_id": c["chunk_id"], "is_relevant": supported and PRESET_FACT in c["text"],
                                 "reason": "MOCK: scripted preset match only"} for c in chunks]}
    elif schema == "StructuredAnswer":
        chunks = json.loads(prompt.rsplit("Context: ", 1)[1])
        found = next((c for c in chunks if supported and PRESET_FACT in c["text"]), None)
        data = {"requested_parts": [query], "unanswered_parts": [] if found else [query],
                "claims": [{"claim_id": "demo-rate", "part": query, "text": PRESET_FACT,
                            "cited_chunk_ids": [found["chunk_id"]]}] if found else []}
    elif schema == "ClaimAudit":
        draft = json.JSONDecoder().raw_decode(prompt.split("\nDraft: ", 1)[1])[0]
        chunks = {c["chunk_id"]: c for c in json.loads(prompt.rsplit("Context: ", 1)[1])}
        evaluations = []
        for claim in draft["claims"]:
            quotes = [{"chunk_id": identity, "excerpt": PRESET_FACT}
                      for identity in claim["cited_chunk_ids"]
                      if identity in chunks and PRESET_FACT in chunks[identity]["text"]]
            valid = supported and claim["text"] == PRESET_FACT and bool(quotes)
            evaluations.append({"claim_id": claim["claim_id"], "verdict": "supported" if valid else "insufficient_evidence",
                "evidence": quotes, "conflicting_evidence": [], "explanation": "MOCK: scripted literal match, not model judgment"})
        answered = bool(evaluations) and all(e["verdict"] == "supported" for e in evaluations)
        data = {"evaluations": evaluations, "part_evaluations": [{"part": part, "is_answered": answered,
                "explanation": "MOCK preset"} for part in draft["requested_parts"]],
                "unanswered_parts": [] if answered else draft["requested_parts"]}
    else:
        raise ValueError("Unexpected demo schema")
    return NS(choices=[NS(message=NS(content=json.dumps(data)))], usage=None)


@contextmanager
def demo_mock_providers():
    with mock_providers():  # Local hash embeddings, no credentials, Python network blocked.
        config._client("openai").chat.completions.create = _mock_generation
        yield


def run_demo(question, selected_documents, pipeline="verified", *, live=False, provider="Gemini"):
    """Return only safe display data; never print exceptions or SDK request bodies."""
    started = time.perf_counter()
    result = {"outcome": "error", "answer": None, "error": None, "chunks": [], "arm": None,
              "indexing": None, "retrieval": None, "query_cost": usage([]),
              "query_latency_ms": 0, "mock": not live, "question": question,
              "selected_documents": list(selected_documents), "pipeline_mode": pipeline}
    files = sample_documents()
    if not question.strip() or not selected_documents or any(name not in files for name in selected_documents):
        result["error"] = "Enter a question and select at least one included document."
        return result
    if pipeline not in ("basic", "verified") or provider not in LIVE_MODELS:
        result["error"] = "Choose a supported pipeline and provider."
        return result
    embedding, generation = LIVE_MODELS[provider] if live else (MOCK_EMBEDDING, "openai/mock-demo-v1")
    result.update(embedding_model=embedding, generation_model=generation)
    try:
        with _DEMO_LOCK, ExitStack() as stack:
            fallback = config.GENERATION_FALLBACK_ENABLED.set(False)
            stack.callback(config.GENERATION_FALLBACK_ENABLED.reset, fallback)
            if not live:
                stack.enter_context(patch.object(config, "CHROMA_PERSIST_DIR", str(Path(config.CHROMA_PERSIST_DIR) / "demo_mock")))
                stack.enter_context(patch.object(config, "CHROMA_CLIENT", None))
                stack.enter_context(demo_mock_providers())
            docs, result["indexing"] = measure(lambda: [ingest_document(str(files[name]), embedding, 500, 100)
                                                         for name in dict.fromkeys(selected_documents)])
            if result["indexing"]["error"]:
                result["error"] = "Indexing failed. Check local credentials, model availability, or index configuration."
                return result
            chunks, result["retrieval"] = measure(lambda: retrieve(question, [doc.id for doc in docs], embedding, 500, 100, 3))
            if result["retrieval"]["error"]:
                result["error"] = "Retrieval failed. Check the selected embedding provider and index configuration."
                result["query_cost"] = usage(result["retrieval"]["calls"])
                result["query_latency_ms"] = result["retrieval"]["latency_ms"]
                return result
            result["chunks"] = [c.model_dump() for c in chunks]
            arm = run_arm(pipeline, question, chunks, generation, embedding)
            result["arm"] = arm
            result["query_cost"] = usage(result["retrieval"]["calls"] + arm["generation_cost"]["calls"])
            result["query_latency_ms"] = result["retrieval"]["latency_ms"] + arm["generation_cost"]["latency_ms"]
            if arm["error"]:
                result["error"] = "The model request failed. Check provider access and retry later. No answer was accepted."
            else:
                result["outcome"], result["answer"] = arm["outcome"], arm["answer"]
    except Exception:
        # Streamlit must never show a raw exception, which may contain credentials or request data.
        result["error"] = "The demo could not complete this request. Check local configuration."
    finally:
        result["total_latency_ms"] = (time.perf_counter() - started) * 1000
    return result
