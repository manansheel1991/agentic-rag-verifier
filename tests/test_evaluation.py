"""Evaluation mechanics only. Scripted fixtures do not establish model performance."""
import copy
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from src import config, evaluation, main
from src.config import ProviderError
from src.evaluation import DEFAULT_DATASET, load_dataset, run_arm, run_evaluation, usage
from src.evaluation_review import read_review, report, summarize, write_review
from src.schemas import BatchContextVerification, ClaimAudit, ChunkMetadata, StructuredAnswer


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    # run_evaluation itself replaces providers and blocks sockets in mock mode.
    output = tmp_path_factory.mktemp("evaluation") / "mock"
    artifact = run_evaluation(DEFAULT_DATASET, output, mode="mock")
    return artifact, output


def test_development_data_integrity():
    data, files, hashes = load_dataset(DEFAULT_DATASET)
    assert len(data.questions) == 25 and len(files) == 6
    assert set(Counter(q.category for q in data.questions).values()) == {5}
    assert Counter(q.answerability for q in data.questions) == {"answerable": 15, "unanswerable": 5, "conflicted": 5}
    assert data.review_required and data.license == "CC0-1.0"
    assert "not an independent benchmark" in data.description
    assert all(q.required_facts and q.reference_answer for q in data.questions)
    assert len(hashes["sha256"]) == len(hashes["dataset_sha256"]) == 64
    assert all(len(q.gold_evidence) >= 2 for q in data.questions if q.category in ("multi_passage", "conflict"))


def test_mock_artifacts_cost_accounting_and_no_auto_scores(mock_run):
    data, output = mock_run
    assert data["mode"] == "mock" and "NOT model performance" in data["notice"]
    assert data["complete"] and len(data["questions"]) == 25
    assert data["configuration"]["generation_temperature"] == 0
    assert data["configuration"]["generation_fallback"] is False
    assert data["configuration"]["recovery"] is False
    assert data["configuration"]["embedding_model"] == "mock/hash-v1"
    assert data["indexing"]["model_calls"] == 6
    for pair in data["questions"]:
        assert pair["basic"]["initial_retrieved_chunk_ids"] == pair["verified"]["initial_retrieved_chunk_ids"]
        assert pair["basic"]["initial_retrieved_chunk_ids"] == [c["chunk_id"] for c in pair["retrieved_evidence"]]
        assert pair["shared_retrieval"]["model_calls"] == 1
        assert pair["basic"]["generation_cost"]["model_calls"] == 1
        assert pair["verified"]["generation_cost"]["model_calls"] == 3
        assert pair["basic"]["error"] is None and pair["verified"]["error"] is None
        for arm in ("basic", "verified"):
            assert pair[arm]["attributed_query_cost"]["available_total_tokens"] is None
            assert pair[arm]["latency_ms"] >= pair["shared_retrieval"]["latency_ms"]
    summary = json.loads((output / "comparison.json").read_text())
    assert summary["physical_run_cost"]["model_calls"] == 131  # 6 indexing + 25 retrieval + 100 generation.
    assert summary["configurations"]["basic"]["attributed_query_cost"]["model_calls"] == 50
    assert summary["configurations"]["verified"]["attributed_query_cost"]["model_calls"] == 100
    for arm in summary["configurations"].values():
        assert all(metric["value"] is None for metric in arm["quality"].values())
    assert data["code_version"]["source_sha256"] and data["code_version"]["git_commit"]
    assert (output / "index" / "documents.sqlite3").exists()
    assert "unscored" in (output / "comparison.md").read_text()


def test_review_is_side_by_side_and_initially_blank(mock_run):
    data, output = mock_run
    labels = read_review(data, output / "review.csv")
    assert len(labels) == 25
    first = next(iter(labels.values()))
    assert "basic_answer" in first and "verified_answer" in first
    assert first["reference_answer"] and first["gold_evidence"] and first["required_facts"]
    assert first["basic_correctness"] == first["verified_correctness"] == ""
    assert first["basic_unsupported_claims"] == first["verified_unsupported_claims"] == ""


def test_same_generator_format_and_no_basic_verification(monkeypatch):
    chunk = ChunkMetadata(chunk_id="stable", document_id="doc", file_name="test.md", page_number=None,
                          text="A fact.", char_offset=0, char_length=7)
    calls = []
    draft = {"requested_parts": ["question"], "claims": [{"claim_id": "c", "part": "question",
             "text": "A fact.", "cited_chunk_ids": ["stable"]}], "unanswered_parts": []}
    def provider(prompt, schema, **kwargs):
        calls.append((schema, prompt, kwargs))
        if schema is StructuredAnswer:
            return json.dumps(draft)
        if schema is BatchContextVerification:
            return json.dumps({"evaluations": [{"chunk_id": "stable", "is_relevant": True, "reason": "mock"}]})
        assert schema is ClaimAudit
        return json.dumps({"evaluations": [{"claim_id": "c", "verdict": "supported",
            "evidence": [{"chunk_id": "stable", "excerpt": "A fact."}], "conflicting_evidence": [], "explanation": "mock"}],
            "part_evaluations": [{"part": "question", "is_answered": True, "explanation": "mock"}], "unanswered_parts": []})
    monkeypatch.setattr(config, "generate_structured_json", provider)
    basic = run_arm("basic", "question", [chunk], "openai/same-model", "same-embedding")
    assert [schema for schema, _, _ in calls] == [StructuredAnswer]
    verified = run_arm("verified", "question", [chunk], "openai/same-model", "same-embedding")
    gen_calls = [(prompt, args) for schema, prompt, args in calls if schema is StructuredAnswer]
    assert gen_calls[0] == gen_calls[1]  # Same evidence order, prompt, model and temperature.
    assert gen_calls[0][1] == {"temperature": 0.0, "model_name": "openai/same-model"}
    assert basic["answer"] == verified["answer"] == "A fact. [test.md; stable]"
    assert basic["emitted_claims"] == verified["emitted_claims"]


def test_uncited_basic_claim_is_retained_for_human_review(monkeypatch):
    monkeypatch.setattr(config, "generate_structured_json", lambda *a, **k: json.dumps({
        "requested_parts": ["q"], "claims": [{"claim_id": "c", "part": "q", "text": "True but uncited.", "cited_chunk_ids": []}],
        "unanswered_parts": []}))
    result = run_arm("basic", "q", [], "openai/model", "embedding")
    assert result["error"] is None and result["answer"] == "True but uncited."
    assert result["emitted_claims"][0]["cited_chunk_ids"] == []


def test_metrics_denominators_partial_errors_and_uncited_claims(mock_run):
    data, output = mock_run
    artifact = copy.deepcopy(data)
    labels = read_review(artifact, output / "review.csv")
    direct, absent = "direct-01", "absent-01"
    # An incomplete output can contain supported facts: correctness and unsupportedness are distinct.
    labels[direct].update(basic_correctness="incorrect", basic_completeness="partial",
                         basic_factual_claims="2", basic_unsupported_claims="1", basic_citation_supported_claims="1", basic_abstention="no")
    labels[absent].update(basic_correctness="correct", basic_completeness="complete",
                         basic_factual_claims="0", basic_unsupported_claims="0", basic_citation_supported_claims="0", basic_abstention="yes")
    # Include one operational error as a reviewed incorrect output; never treat it as an abstention.
    labels["direct-02"].update(basic_correctness="incorrect", basic_abstention="no")
    artifact["questions"][1]["basic"]["error"] = {"category": "authentication"}
    # A correct uncited factual claim is included in the denominator, not auto-counted as unsupported.
    labels[direct].update(verified_correctness="correct", verified_factual_claims="1",
                         verified_unsupported_claims="0", verified_citation_supported_claims="0", verified_abstention="no")
    metrics = summarize(artifact, labels)["configurations"]
    a, b = metrics["basic"]["quality"], metrics["verified"]["quality"]
    assert a["answer_correctness"]["value"] == 1 / 3
    assert a["unsupported_claim_rate"]["value"] == 1 / 2
    assert a["abstention_on_unanswerable"]["value"] == 1
    assert a["abstention_on_unanswerable"]["eligible_questions"] == 5
    assert a["false_refusals_on_answerable"]["denominator"] == 2
    assert a["false_refusals_on_answerable"]["eligible_questions"] == 15
    assert a["completeness"]["value"] == 1 / 2
    assert b["answer_correctness"]["value"] == 1
    assert b["unsupported_claim_rate"]["value"] == 0
    assert b["citation_support"]["value"] == 0
    assert b["abstention_on_unanswerable"]["value"] is None


def test_zero_claim_denominator_is_unscored_not_zero(mock_run):
    data, output = mock_run
    labels = read_review(data, output / "review.csv")
    labels["absent-01"].update(basic_factual_claims="0", basic_unsupported_claims="0")
    metric = summarize(data, labels)["configurations"]["basic"]["quality"]["unsupported_claim_rate"]
    assert metric["value"] is None and metric["denominator"] == 0 and metric["reviewed_questions"] == 1


@pytest.mark.parametrize("bad", ["duplicate", "wrong_run", "negative", "too_many", "edited_answer"])
def test_invalid_review_rejected(mock_run, tmp_path, bad):
    data, output = mock_run
    with (output / "review.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if bad == "duplicate":
        rows.append(dict(rows[0]))
    elif bad == "wrong_run":
        rows[0]["run_id"] = "different"
    elif bad == "negative":
        rows[0]["basic_factual_claims"] = "-1"
    elif bad == "too_many":
        rows[0].update(basic_factual_claims="1", basic_unsupported_claims="2")
    else:
        rows[0]["basic_answer"] = "changed after the run"
    path = tmp_path / "review.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        read_review(data, path)


def test_attempt_telemetry_includes_failures_and_available_usage(monkeypatch):
    class Transient(Exception):
        status_code = 503
    request = Mock(side_effect=[Transient(), NS(data=[NS(embedding=[1., 2., 3.])], usage=NS(total_tokens=9))])
    monkeypatch.setattr(config, "_client", lambda _: NS(embeddings=NS(create=request)))
    _, measured = evaluation.measure(lambda: config.get_vector_embedding("text", "openai/" + config.OPENAI_EMBED_MODEL))
    assert measured["model_calls"] == 2 and measured["available_total_tokens"] == 9
    assert measured["calls_without_usage"] == 1
    assert measured["calls"][0]["error_category"] == "transient"
    assert measured["error"] is None


def test_evaluation_can_disable_generation_fallback(monkeypatch):
    route = Mock(side_effect=ProviderError("authentication", "gemini"))
    monkeypatch.setattr(config, "_client", route)
    token = config.GENERATION_FALLBACK_ENABLED.set(False)
    try:
        with pytest.raises(ProviderError):
            config.generate_text("prompt")
    finally:
        config.GENERATION_FALLBACK_ENABLED.reset(token)
    route.assert_called_once_with("gemini")


def test_failed_arm_is_recorded_as_error_not_normal_abstention(monkeypatch):
    monkeypatch.setattr(main, "generate_answer", Mock(side_effect=ProviderError("authentication", "openai")))
    value = run_arm("basic", "q", [], "openai/model", "embedding")
    assert value["error"]["category"] == "authentication"
    assert value["answer"] is None and value["emitted_claims"] == []


def test_results_and_reviews_cannot_be_overwritten(mock_run):
    _, output = mock_run
    before = (output / "review.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_evaluation(DEFAULT_DATASET, output, mode="mock")
    assert (output / "review.csv").read_bytes() == before


def test_uncited_claim_is_still_blocked_by_verified_pipeline(monkeypatch):
    chunk = ChunkMetadata(chunk_id="stable", document_id="doc", file_name="test.md",
                          text="A fact.", char_offset=0, char_length=7)
    def provider(prompt, schema, **kwargs):
        if schema is BatchContextVerification:
            value = {"evaluations": [{"chunk_id": "stable", "is_relevant": True, "reason": "mock"}]}
        elif schema is StructuredAnswer:
            value = {"requested_parts": ["q"], "claims": [{"claim_id": "c", "part": "q", "text": "A fact.", "cited_chunk_ids": []}], "unanswered_parts": []}
        else:
            value = {"evaluations": [{"claim_id": "c", "verdict": "supported", "evidence": [],
                "conflicting_evidence": [], "explanation": "mock overconfidence"}],
                "part_evaluations": [{"part": "q", "is_answered": True, "explanation": "mock"}], "unanswered_parts": []}
        return json.dumps(value)
    monkeypatch.setattr(config, "generate_structured_json", provider)
    result = run_arm("verified", "q", [chunk], "openai/model", "embedding")
    assert result["answer"] is None and result["emitted_claims"] == []
    assert "Missing chunk citations" in result["pipeline"]["claims"][0]["checks"]["errors"]


def test_report_command_runs_offline(mock_run, tmp_path, monkeypatch):
    import subprocess
    import sys
    _, source = mock_run
    for name in ("results.json", "review.csv"):
        shutil.copy(source / name, tmp_path / name)
    completed = subprocess.run([sys.executable, "-m", "src.evaluation", "report", "--results",
        str(tmp_path / "results.json"), "--review", str(tmp_path / "review.csv")],
        cwd=evaluation.ROOT, text=True, capture_output=True, timeout=30, check=True)
    assert "Comparison saved" in completed.stdout
    assert "MOCK" in (tmp_path / "comparison.md").read_text()
    assert "unscored" in (tmp_path / "comparison.md").read_text()


def test_corpus_hash_detects_source_changes_and_gold_spans_are_validated(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    shutil.copy(DEFAULT_DATASET, dataset_path)
    shutil.copytree(DEFAULT_DATASET.parent / "corpus", tmp_path / "corpus")
    _, _, before = load_dataset(dataset_path)
    path = tmp_path / "corpus" / "harbor-service.md"
    path.write_text(path.read_text() + "\nAn additional synthetic note.\n")
    _, _, after = load_dataset(dataset_path)
    assert before["sha256"] != after["sha256"]
    assert before["dataset_sha256"] == after["dataset_sha256"]
    path.write_text("Replaced content invalidates the gold offsets.")
    with pytest.raises(ValueError, match="gold evidence"):
        load_dataset(dataset_path)
