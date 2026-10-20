"""Small paired development evaluation. No automatic model-based scoring."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict
from src import config, main
from src.ingestion import get_or_create_collection, ingest_document, retrieve, validate_chunking
from src.schemas import PipelineResult
from src.verification import render_claims

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evaluation" / "dataset.json"
ARMS = ("basic", "verified")


class GoldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str
    excerpt: str
    char_start: int
    char_end: int
    page_number: int | None


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: str
    question: str
    answerability: Literal["answerable", "unanswerable", "conflicted"]
    reference_answer: str
    required_facts: list[str]
    gold_evidence: list[GoldEvidence]


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    review_required: bool
    license: str
    documents: list[str]
    questions: list[Question]


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def load_dataset(path):
    path = Path(path).resolve()
    dataset = Dataset.model_validate_json(path.read_text())
    if not dataset.documents or not dataset.questions:
        raise ValueError("Dataset must contain documents and questions")
    if len({q.id for q in dataset.questions}) != len(dataset.questions):
        raise ValueError("Duplicate question IDs")
    corpus = path.parent / "corpus"
    files = {}
    for name in dataset.documents:
        file = (corpus / name).resolve()
        if file.parent != corpus.resolve() or not file.is_file():
            raise ValueError("Corpus document must be a local file inside corpus/")
        files[name] = file
    for q in dataset.questions:
        for gold in q.gold_evidence:
            if gold.document not in files:
                raise ValueError("Gold evidence must reference a selected corpus document")
            text = files[gold.document].read_text(encoding="utf-8")
            if not gold.excerpt or gold.char_start < 0 or text[gold.char_start:gold.char_end] != gold.excerpt:
                raise ValueError("Invalid gold evidence offsets or excerpt")
    hashes = {name: _digest(file.read_bytes()) for name, file in sorted(files.items())}
    return dataset, files, {"files": hashes, "sha256": _digest(json.dumps(hashes, sort_keys=True).encode()),
                            "dataset_sha256": _digest(path.read_bytes())}


def code_version():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        commit, status = None, "unavailable"
    # Includes uncommitted/untracked Python code, prompts, and dependency declarations.
    files = sorted((ROOT / "src").glob("*.py")) + sorted(ROOT.glob("requirements*.txt"))
    hashes = {str(p.relative_to(ROOT)): _digest(p.read_bytes()) for p in files}
    return {"git_commit": commit, "dirty": bool(status), "source_sha256": _digest(
        json.dumps(hashes, sort_keys=True).encode()), "files": hashes}


def usage(events):
    known = [event["tokens"] for event in events if event["tokens"] is not None]
    return {"model_calls": len(events), "available_total_tokens": sum(known) if known else None,
            "calls_with_usage": len(known), "calls_without_usage": len(events) - len(known)}


def measure(call):
    events = []
    attempt_token = config.API_CALL_TRACE.set(events)
    response_token = config.CALL_TRACE.set([])
    started = time.perf_counter()
    value, error = None, None
    try:
        value = call()
    except Exception as exc:
        error = {"category": exc.category if isinstance(exc, config.ProviderError) else type(exc).__name__,
                 "message": str(exc) if isinstance(exc, config.ProviderError) else "Evaluation phase failed"}
    finally:
        config.CALL_TRACE.reset(response_token)
        config.API_CALL_TRACE.reset(attempt_token)
    return value, {"latency_ms": (time.perf_counter() - started) * 1000,
                   "calls": events, **usage(events), "error": error}


def run_arm(name, question, chunks, generation_model, embedding_model):
    result = PipelineResult(outcome="abstained", embedding_model_used=embedding_model,
                            unanswered_parts=[question])
    draft = None
    def stage(label, call):
        started = time.perf_counter()
        try:
            return call()
        finally:
            result.timings[label + "_ms"] = (time.perf_counter() - started) * 1000
    def execute():
        nonlocal draft
        if name == "verified":
            draft = main.answer_from_evidence(question, chunks, generation_model, result, stage)
        else:
            draft = stage("generation", lambda: main.generate_answer(question, chunks, generation_model))
            result.answer = render_claims(draft.claims, chunks)  # No factual/citation verification in A.
            result.requested_parts = draft.requested_parts
            result.unanswered_parts = list(dict.fromkeys(draft.unanswered_parts + [
                part for part in draft.requested_parts if not any(c.part == part for c in draft.claims)]))
            result.outcome = "abstained" if not result.answer else "partial" if result.unanswered_parts else "answered"
        return result
    _, telemetry = measure(execute)
    if telemetry["error"]:
        result.outcome, result.answer = "abstained", None
        result.error_category = telemetry["error"]["category"]
        result.error_message = telemetry["error"]["message"]
    result.model_identities = list(dict.fromkeys(event["model"] for event in telemetry["calls"]))
    if result.model_identities and result.generation_model_used is None:
        result.generation_model_used = result.model_identities[-1]
    emitted = (draft.claims if name == "basic" and draft else [
        item.claim for item in result.claims if item.verdict == "supported" and item.checks.passed])
    if telemetry["error"]:
        emitted = []
    return {"answer": result.answer, "outcome": result.outcome,
            "emitted_claims": [claim.model_dump() for claim in emitted],
            "draft": draft.model_dump() if draft else None, "pipeline": result.model_dump(),
            "generation_cost": telemetry, "error": telemetry["error"]}


def run_evaluation(dataset_path, output, *, mode, embedding_model=None, generation_model=None,
                   chunk_size=500, overlap=100, top_k=3):
    if mode not in ("mock", "live"):
        raise ValueError("Choose mock or live explicitly")
    validate_chunking(chunk_size, overlap)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    dataset, files, corpus_hash = load_dataset(dataset_path)
    if mode == "mock":
        from src.evaluation_mock import MOCK_EMBEDDING, MOCK_GENERATION
        embedding_model, generation_model = MOCK_EMBEDDING, MOCK_GENERATION
    else:
        embedding_model = embedding_model or config.GEMINI_EMBED_MODEL
        generation_model = generation_model or config.GEMINI_GEN_MODEL
        config.embedding_spec(embedding_model)  # Reject unsupported identities before indexing costs.
    provider, model = generation_model.split("/", 1) if "/" in generation_model else (
        "gemini" if generation_model.startswith("gemini-") else "openai", generation_model)
    if provider not in ("gemini", "openai") or not model:
        raise ValueError("Invalid generation model identity")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # Do not overwrite existing review labels or indexes.
    artifact = {"schema_version": 1, "complete": False, "run_id": str(uuid.uuid4()), "mode": mode,
        "notice": "MOCK: scripted plumbing output, NOT model performance" if mode == "mock" else
                  "Synthetic development data requiring review; NOT an independent benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "code_version": code_version(),
        "runtime": {"python": sys.version.split()[0], "packages": {
            name: importlib.metadata.version(name) for name in ("chromadb", "pydantic", "openai", "google-genai")}},
        "corpus": corpus_hash, "dataset": dataset.model_dump(), "configuration": {
            "embedding_model": embedding_model, "generation_model": generation_model,
            "generation_identity": f"{provider}/{model}",
            "chunk_size": chunk_size, "overlap": overlap, "top_k": top_k,
            "generation_temperature": 0.0, "generation_schema": "StructuredAnswer",
            "max_output_tokens": "provider default, identical for both arms",
            "generation_fallback": False, "recovery": False, "max_attempts": config.MAX_ATTEMPTS,
            "api_timeout_seconds": config.API_TIMEOUT_SECONDS,
            "initial_retrieval": "once per question, identical ordered chunks for A and B",
            "arm_order": "alternates basic-first and verified-first by question index"}, "questions": []}
    with ExitStack() as stack:
        stack.enter_context(patch.object(config, "CHROMA_PERSIST_DIR", str(output / "index")))
        stack.enter_context(patch.object(config, "CHROMA_CLIENT", None))
        fallback_token = config.GENERATION_FALLBACK_ENABLED.set(False)
        stack.callback(config.GENERATION_FALLBACK_ENABLED.reset, fallback_token)
        if mode == "mock":
            from src.evaluation_mock import mock_providers
            stack.enter_context(mock_providers())
        docs, indexing = measure(lambda: [ingest_document(str(path), embedding_model, chunk_size, overlap)
                                         for path in files.values()])
        artifact["indexing"] = indexing
        artifact["indexed_documents"] = [doc.model_dump() for doc in docs] if docs else []
        artifact["configuration"]["embedding_identity"] = config.embedding_spec(embedding_model).model_dump()
        artifact["configuration"]["collection_names"] = sorted({d.collection_name for d in docs or []})
        artifact["configuration"]["index_metadata"] = (get_or_create_collection(
            embedding_model, chunk_size, overlap).metadata if docs else None)
        (output / "results.json").write_text(json.dumps(artifact, indent=2) + "\n")
        for index, q in enumerate(dataset.questions):
            if indexing["error"]:
                chunks, retrieval_cost = [], {"latency_ms": 0, "calls": [], **usage([]), "error": indexing["error"]}
            else:
                chunks, retrieval_cost = measure(lambda: retrieve(q.question, [d.id for d in docs],
                    embedding_model, chunk_size, overlap, top_k))
                chunks = chunks or []
            row = {"id": q.id, "shared_retrieval": retrieval_cost,
                   "retrieved_evidence": [c.model_dump() for c in chunks]}
            for name in ARMS if index % 2 == 0 else reversed(ARMS):
                if retrieval_cost["error"]:
                    arm = {"answer": None, "outcome": "abstained", "emitted_claims": [],
                           "draft": None, "pipeline": None, "generation_cost": {
                               "latency_ms": 0, "calls": [], **usage([]), "error": None},
                           "error": retrieval_cost["error"]}
                else:
                    arm = run_arm(name, q.question, chunks, generation_model, embedding_model)
                arm["latency_ms"] = retrieval_cost["latency_ms"] + arm["generation_cost"]["latency_ms"]
                arm["initial_retrieved_chunk_ids"] = [chunk.chunk_id for chunk in chunks]
                arm["attributed_query_cost"] = usage(retrieval_cost["calls"] + arm["generation_cost"]["calls"])
                row[name] = arm
            artifact["questions"].append(row)
            # Persist progress after each pair; interrupted runs remain identifiable as incomplete.
            artifact["complete"] = len(artifact["questions"]) == len(dataset.questions)
            (output / "results.json").write_text(json.dumps(artifact, indent=2) + "\n")
    from src.evaluation_review import write_review, report
    write_review(artifact, output / "review.csv")
    report(output / "results.json", output / "review.csv")
    return artifact


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run paired configurations; live uses paid providers")
    run.add_argument("--mode", choices=["mock", "live"], required=True)
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--embedding-model")
    run.add_argument("--generation-model")
    run.add_argument("--chunk-size", type=int, default=500)
    run.add_argument("--overlap", type=int, default=100)
    run.add_argument("--top-k", type=int, default=3)
    review = commands.add_parser("report", help="Summarize human review; never calls providers")
    review.add_argument("--results", type=Path, required=True)
    review.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_evaluation(args.dataset, args.output, mode=args.mode, embedding_model=args.embedding_model,
                       generation_model=args.generation_model, chunk_size=args.chunk_size,
                       overlap=args.overlap, top_k=args.top_k)
        print(f"{args.mode.upper()} artifacts saved to {args.output}. Quality metrics await human review.")
    else:
        from src.evaluation_review import report
        report(args.results, args.review)
        print(f"Comparison saved beside {args.results}.")


if __name__ == "__main__":
    cli()
