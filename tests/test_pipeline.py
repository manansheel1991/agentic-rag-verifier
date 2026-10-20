import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import chromadb
import pytest
from chromadb.api.models.Collection import Collection
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from src import config
from src.ingestion import _client
from src.config import GEMINI_EMBED_MODEL as MODEL, OPENAI_EMBED_MODEL as OTHER, ProviderError
from src.ingestion import (ingest_document, get_or_create_collection, retrieve, _get_collection_name,
                           _chunk_text_with_metadata, validate_embedding, delete_document_from_collection)
from src.main import verify_context, run_pipeline
from src.schemas import ChunkMetadata, BatchContextVerification, ClaimAudit, StructuredAnswer


def collection(model=MODEL, size=70, overlap=10):
    return get_or_create_collection(model, size, overlap)


def ingest(path, model=MODEL):
    return ingest_document(str(path), model, 70, 10)


def test_persistence_across_process_restart(document, embed):
    doc = ingest(document)
    expected = collection().count()
    # A different process cannot reuse Chroma's in-process client cache.
    script = '''
import sys
from src import config
from src.ingestion import _client
from src.ingestion import retrieve
config.CHROMA_PERSIST_DIR = sys.argv[1]
config.EMBEDDING_DIMENSIONS[("gemini", config.GEMINI_EMBED_MODEL)] = 3
config.get_vector_embedding = lambda *a, **k: [1., .2, .5]
chunks = retrieve("query", sys.argv[2], config.GEMINI_EMBED_MODEL, 70, 10)
assert chunks and all(c.document_id == sys.argv[2] for c in chunks)
print("persisted")
'''
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "ANONYMIZED_TELEMETRY": "False"}
    completed = subprocess.run([sys.executable, "-c", script, config.CHROMA_PERSIST_DIR, doc.id],
                               capture_output=True, text=True, check=True, timeout=30, env=env)
    assert "persisted" in completed.stdout
    assert collection().count() == expected


def test_idempotency_touch_and_stable_ids(document, embed):
    first = ingest(document)
    ids = collection().get()["ids"]
    calls = len(embed)
    os.utime(document, None)
    second = ingest(document)
    assert first.id == second.id
    assert collection().get()["ids"] == ids
    assert len(embed) == calls


def test_document_update_removes_old_chunks(document, embed):
    first = ingest(document)
    old_ids = set(collection().get()["ids"])
    document.write_text("Replacement text about updates.")
    second = ingest(document)
    assert first.id == second.id
    assert first.file_hash != second.file_hash
    assert not old_ids.intersection(collection().get()["ids"])
    chunks = retrieve("updates", second.id, MODEL, 70, 10)
    assert chunks and all(c.file_hash == second.file_hash for c in chunks)


def test_failed_embedding_update_preserves_old_version(document, embed, monkeypatch):
    first = ingest(document)
    old_ids = collection().get()["ids"]
    document.write_text("new version " * 20)
    monkeypatch.setattr(config, "get_vector_embedding", Mock(side_effect=ProviderError("rate_limit", "gemini")))
    with pytest.raises(ProviderError):
        ingest(document)
    assert collection().get()["ids"] == old_ids
    monkeypatch.setattr(config, "get_vector_embedding", lambda *a, **k: [1, 2, 3])
    assert all(c.file_hash == first.file_hash for c in retrieve("q", first.id, MODEL, 70, 10))


def test_failed_partial_chroma_update_never_activates(document, embed, monkeypatch):
    first = ingest(document)
    old_ids = collection().get()["ids"]
    document.write_text("new version " * 20)
    original = Collection.upsert
    def fail_after_write(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("simulated storage failure")
    with monkeypatch.context() as patch:
        patch.setattr(Collection, "upsert", fail_after_write)
        with pytest.raises(RuntimeError):
            ingest(document)
    assert collection().get()["ids"] == old_ids
    assert all(c.file_hash == first.file_hash for c in retrieve("q", first.id, MODEL, 70, 10))
    assert ingest(document).file_hash != first.file_hash


def test_staging_is_inactive_even_if_cleanup_fails(document, embed, monkeypatch):
    first = ingest(document)
    document.write_text("replacement " * 15)
    with monkeypatch.context() as patch:
        patch.setattr(Collection, "delete", Mock(side_effect=RuntimeError("cleanup failed")))
        second = ingest(document)
    assert collection().count() > len(collection().get(where={"file_hash": second.file_hash})["ids"])
    chunks = retrieve("q", first.id, MODEL, 70, 10, 100)
    assert chunks and all(c.file_hash == second.file_hash for c in chunks)


def test_document_isolation_and_multiple_selected(document, embed, tmp_path):
    first = ingest(document)
    other = tmp_path / "other.txt"
    other.write_text("Secret unrelated information " * 10)
    second = ingest(other)
    only = retrieve("query", first.id, MODEL, 70, 10, 100)
    assert only and {c.document_id for c in only} == {first.id}
    both = retrieve("query", [first.id, second.id], MODEL, 70, 10, 100)
    assert {c.document_id for c in both} == {first.id, second.id}
    assert retrieve("q", [], MODEL, 70, 10) == []
    assert retrieve("q", "missing", MODEL, 70, 10) == []


def test_metadata_and_offsets(document, embed):
    doc = ingest(document)
    chunks = retrieve("q", doc.id, MODEL, 70, 10, 100)
    original = document.read_text()
    for c in chunks:
        assert c.text == original[c.char_offset:c.char_offset + c.char_length]
        assert c.file_name == document.name
        assert c.file_path == str(document.resolve())
        assert c.file_hash == doc.file_hash
        assert c.chunk_text_hash == hashlib.sha256(c.text.encode()).hexdigest()
        assert c.model_dump()["text"] == c.text
        assert c.page_number is None


def make_pdf(path):
    writer = PdfWriter()
    texts = ["First page text about reliability.", "Second page text about persistence."]
    for text in texts:
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)
    return texts


def test_real_pdf_page_metadata(tmp_path, embed):
    path = tmp_path / "pages.pdf"
    texts = make_pdf(path)
    doc = ingest(path)
    chunks = retrieve("q", doc.id, MODEL, 70, 10, 100)
    assert doc.page_count == 2
    assert {c.page_number for c in chunks} == {1, 2}
    for c in chunks:
        assert c.text == texts[c.page_number - 1][c.char_offset:c.char_offset + c.char_length]


def test_separate_model_and_chunking_collections(document, embed):
    first = ingest(document)
    second = ingest(document, OTHER)  # Same dimensions; still a separate provider/model.
    assert first.collection_name != second.collection_name
    assert collection().metadata["hnsw:space"] == "cosine"
    assert collection().metadata["dimensions"] == 3
    assert collection().metadata["embedding_provider"] == "gemini"
    assert collection().metadata["index_schema_version"] == 2
    assert _get_collection_name(MODEL, 71, 10) != first.collection_name
    assert _get_collection_name("gemini/" + MODEL, 70, 10) == first.collection_name
    retrieve("q", first.id, MODEL, 70, 10)
    assert embed[-1][1] == MODEL
    retrieve("q", second.id, "openai/" + OTHER, 70, 10)
    assert embed[-1][1] == "openai/" + OTHER


@pytest.mark.parametrize("key,value", [("embedding_provider", "openai"), ("embedding_model", OTHER),
    ("dimensions", 42), ("chunk_size", 99), ("chunk_overlap", 1), ("index_schema_version", 99), ("hnsw:space", "l2")])
def test_collection_metadata_mismatch_is_error(key, value):
    current = collection()
    metadata = {**current.metadata, key: value}
    client = _client()
    client.delete_collection(current.name)
    client.create_collection(current.name, embedding_function=None, metadata=metadata)
    with pytest.raises(ValueError, match="metadata mismatch"):
        collection()


@pytest.mark.parametrize("vector", [[float("nan"), 1, 1], [float("inf"), 1, 1], [0, 0, 0],
                                      [1, 2], [[1], 2, 3], ["1", 2, 3], None])
def test_invalid_embeddings_rejected(vector, document, monkeypatch):
    monkeypatch.setattr(config, "get_vector_embedding", lambda *a, **k: vector)
    with pytest.raises(ValueError):
        ingest(document)
    assert collection().count() == 0


def test_invalid_query_embedding(document, embed, monkeypatch):
    doc = ingest(document)
    monkeypatch.setattr(config, "get_vector_embedding", lambda *a, **k: [0, 0, 0])
    with pytest.raises(ValueError, match="norm"):
        retrieve("q", doc.id, MODEL, 70, 10)


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11), (1.5, 0)])
def test_invalid_chunking_even_with_empty_input(size, overlap):
    with pytest.raises(ValueError, match="0 <= overlap < chunk_size"):
        _chunk_text_with_metadata("", "id", size, overlap)


@pytest.mark.parametrize("content", ["", "  \n\t"])
def test_empty_update_keeps_previous(document, embed, content):
    first = ingest(document)
    document.write_text(content)
    with pytest.raises(ValueError, match="no readable text"):
        ingest(document)
    assert all(c.file_hash == first.file_hash for c in retrieve("q", first.id, MODEL, 70, 10))


def test_unreadable_documents(tmp_path, embed):
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        ingest(missing)
    missing.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        ingest(missing)
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a PDF")
    with pytest.raises(Exception):
        ingest(broken)
    assert not embed


def chunks():
    return [ChunkMetadata(chunk_id=i, document_id="doc", char_offset=0, char_length=4, text="text")
            for i in ("a", "b")]


@pytest.mark.parametrize("ids", [["a", "unknown"], ["a", "a", "b"], ["a"], []])
def test_critic_strict_validation(ids, monkeypatch):
    monkeypatch.setattr(config, "generate_structured_json", lambda *a, **k: json.dumps({
        "evaluations": [{"chunk_id": identity, "is_relevant": True, "reason": "reason"} for identity in ids]}))
    with pytest.raises(ProviderError, match="malformed_output"):
        verify_context("q", chunks())


def test_critic_valid_and_malformed_output(monkeypatch):
    monkeypatch.setattr(config, "generate_structured_json", lambda *a, **k: json.dumps({"evaluations": [
        {"chunk_id": "b", "is_relevant": False, "reason": "no"},
        {"chunk_id": "a", "is_relevant": True, "reason": "yes"}]}))
    assert [c.chunk_id for c in verify_context("q", chunks())] == ["a"]
    monkeypatch.setattr(config, "generate_structured_json", lambda *a, **k: "not json")
    with pytest.raises(ProviderError):
        verify_context("q", chunks())


def test_deletion(document, embed):
    doc = ingest(document)
    delete_document_from_collection(doc.id, collection())
    assert collection().count() == 0
    assert retrieve("q", doc.id, MODEL, 70, 10) == []
    assert ingest(document).file_hash == doc.file_hash


def fake_generation(monkeypatch, faithful=True, relevant=True):
    def structured(prompt, schema, **kwargs):
        model = kwargs["model_name"]
        tokens = 13 if schema is StructuredAnswer else 11
        config.CALL_TRACE.get().append({"kind": "generation", "model": "openai/" + model, "tokens": tokens})
        if schema is BatchContextVerification:
            rows = json.loads(prompt.split("Available Document Chunks:\n")[1].split("\n")[0])
            return json.dumps({"evaluations": [{"chunk_id": row["chunk_id"], "is_relevant": relevant, "reason": "checked"} for row in rows]})
        rows = json.loads(prompt.split("Context: ")[1])
        if schema is StructuredAnswer:
            return json.dumps({"requested_parts": ["q"], "unanswered_parts": [], "claims": [{
                "claim_id": "c1", "part": "q", "text": "Grounded answer.",
                "cited_chunk_ids": [rows[0]["chunk_id"]]}]})
        return json.dumps({"evaluations": [{"claim_id": "c1",
            "verdict": "supported" if faithful else "insufficient_evidence",
            "evidence": [{"chunk_id": rows[0]["chunk_id"], "excerpt": rows[0]["text"]}],
            "conflicting_evidence": [], "explanation": "audited"}],
            "part_evaluations": [{"part": "q", "is_answered": faithful, "explanation": "checked"}],
            "unanswered_parts": []})
    monkeypatch.setattr(config, "generate_structured_json", structured)


def test_pipeline_typed_result_model_timings_usage(document, embed, monkeypatch):
    fake_generation(monkeypatch)
    result = run_pipeline(str(document), "q", 70, 10, generation_model="custom-model")
    assert result.outcome == "answered"
    assert result.answer.startswith("Grounded answer. [")
    assert result.sources and result.sources[0].model_dump()["text"]
    assert result.generation_model_used == "openai/custom-model"
    assert result.tokens_used_generation == 35
    assert result.tokens_used_embedding is None
    assert all(result.timings[k + "_ms"] >= 0 for k in ["ingestion", "retrieval", "context_verification", "generation", "claim_audit", "total_pipeline"])
    assert config.CALL_TRACE.get() is None


@pytest.mark.parametrize("faithful,relevant,category", [(False, True, None), (True, False, "no_relevant_context")])
def test_pipeline_fail_closed(document, embed, monkeypatch, faithful, relevant, category):
    fake_generation(monkeypatch, faithful, relevant)
    result = run_pipeline(str(document), "q", 70, 10)
    assert result.outcome == "abstained" and result.error_category == category
    assert result.answer != "Grounded answer."


def test_pipeline_provider_failure(document, embed, monkeypatch):
    fake_generation(monkeypatch)
    structured = config.generate_structured_json
    def fail_answer(prompt, schema, **kwargs):
        if schema is StructuredAnswer:
            raise ProviderError("exhausted_providers", "generation", ["gemini:rate_limit", "openai:authentication"])
        return structured(prompt, schema, **kwargs)
    monkeypatch.setattr(config, "generate_structured_json", fail_answer)
    result = run_pipeline(str(document), "q", 70, 10)
    assert result.outcome == "abstained" and result.answer is None
    assert result.error_category == "exhausted_providers"
    assert "generation_ms" in result.timings
    assert "rate_limit" in result.error_message


def test_later_batch_failure_preserves_active_document(document, embed, monkeypatch):
    first = ingest(document)
    old_ids = set(collection().get()["ids"])
    document.write_text("updated text " * 30)
    client_type = type(_client())
    original = Collection.upsert
    calls = []
    def fail_second_batch(self, *args, **kwargs):
        calls.append(kwargs["ids"])
        if len(calls) == 2:
            raise RuntimeError("failed second batch")
        return original(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(client_type, "get_max_batch_size", lambda self: 1)
        patch.setattr(Collection, "upsert", fail_second_batch)
        with pytest.raises(RuntimeError):
            ingest(document)
    assert len(calls) == 2
    assert set(collection().get()["ids"]) == old_ids
    assert all(c.file_hash == first.file_hash for c in retrieve("q", first.id, MODEL, 70, 10))


@pytest.mark.parametrize("mode", ["script", "module", "default"])
def test_cli_offline(document, tmp_path, mode):
    repository = Path(__file__).resolve().parents[1]
    script = '''
import json, runpy, socket, sys
from src import config
from src.schemas import BatchContextVerification, StructuredAnswer

def blocked(*a, **k):
    raise AssertionError("Network or real provider call forbidden")
socket.socket.connect = blocked
config._client = blocked
config.EMBEDDING_DIMENSIONS[("gemini", config.GEMINI_EMBED_MODEL)] = 3
config.get_vector_embedding = lambda *a, **k: [1., 2., 3.]
def structured(prompt, schema, **kwargs):
    if schema is BatchContextVerification:
        rows = json.loads(prompt.split("Available Document Chunks:\\n")[1].split("\\n")[0])
        return json.dumps({"evaluations": [{"chunk_id": row["chunk_id"], "is_relevant": True, "reason": "checked"} for row in rows]})
    rows = json.loads(prompt.split("Context: ")[1])
    if schema is StructuredAnswer:
        return json.dumps({"requested_parts": ["q"], "unanswered_parts": [], "claims": [{
            "claim_id": "c1", "part": "q", "text": "CLI mocked answer", "cited_chunk_ids": [rows[0]["chunk_id"]]}]})
    return json.dumps({"evaluations": [{"claim_id": "c1", "verdict": "supported",
        "evidence": [{"chunk_id": rows[0]["chunk_id"], "excerpt": rows[0]["text"]}],
        "conflicting_evidence": [], "explanation": "checked"}],
        "part_evaluations": [{"part": "q", "is_answered": True, "explanation": "checked"}],
        "unanswered_parts": []})
config.generate_structured_json = structured
mode, path, repo = sys.argv[1:]
sys.argv = ["src/main.py"]
if mode != "default":
    sys.argv += ["-f", path, "-q", "query", "--chunk_size", "70", "--chunk_overlap", "10"]
if mode == "script":
    runpy.run_path(repo + "/src/main.py", run_name="__main__")
else:
    runpy.run_module("src.main", run_name="__main__")
'''
    env = {**os.environ, "PYTHONPATH": str(repository), "CHROMA_PERSIST_DIR": str(tmp_path / "cli-index"),
           "HF_HUB_OFFLINE": "1", "ANONYMIZED_TELEMETRY": "False"}
    completed = subprocess.run([sys.executable, "-c", script, mode, str(document), str(repository)],
                               cwd=tmp_path, env=env, capture_output=True, text=True, check=True, timeout=30)
    assert '"outcome": "answered"' in completed.stdout
    assert "CLI mocked answer" in completed.stdout
    assert '"text":' in completed.stdout


def test_runtime_distance_configuration_is_validated():
    # Even correctly labelled metadata must not hide a different index distance.
    current = collection()
    assert current.configuration["hnsw"]["space"] == "cosine"
    client = _client()
    metadata = dict(current.metadata)
    name = current.name
    client.delete_collection(name)
    metadata.pop("hnsw:space")
    wrong = client.create_collection(name, embedding_function=None,
                                    metadata=metadata, configuration={"hnsw": {"space": "l2"}})
    # Restore the declared cosine metadata without touching Chroma's actual index configuration.
    from chromadb.api.models.Collection import Collection as ChromaCollection
    original = ChromaCollection._validate_modify_request
    try:
        ChromaCollection._validate_modify_request = lambda *a, **k: None
        wrong.modify(metadata={**metadata, "hnsw:space": "cosine"})
    finally:
        ChromaCollection._validate_modify_request = original
    with pytest.raises(ValueError, match="distance configuration mismatch"):
        collection()


def test_query_provider_unavailable_does_not_switch_models(document, embed, monkeypatch):
    doc = ingest(document)
    getter = Mock(side_effect=ProviderError("unavailable", "gemini"))
    monkeypatch.setattr(config, "get_vector_embedding", getter)
    with pytest.raises(ProviderError, match="gemini: unavailable"):
        retrieve("q", doc.id, MODEL, 70, 10)
    getter.assert_called_once_with("q", model_name=MODEL)
