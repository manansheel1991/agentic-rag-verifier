"""Mocked judgments test enforcement, not the accuracy of semantic support decisions."""
import copy
import json
from types import SimpleNamespace

import pytest

from src import config, main
from src.schemas import BatchContextVerification, ClaimAudit, ChunkMetadata, StructuredAnswer


@pytest.fixture
def evidence():
    chunks = [
        ChunkMetadata(chunk_id="stable-a", document_id="doc-a", file_name="budget.pdf", page_number=2,
                      text="The budget is $100. The launch is in June.", char_offset=0, char_length=41),
        ChunkMetadata(chunk_id="stable-b", document_id="doc-b", file_name="revision.pdf", page_number=5,
                      text="The budget is $200.", char_offset=20, char_length=19),
    ]
    for chunk in chunks:
        chunk.char_length = len(chunk.text)
    return chunks


@pytest.fixture
def draft():
    return {"requested_parts": ["What is the budget?"], "unanswered_parts": [], "claims": [
        {"claim_id": "claim-budget", "part": "What is the budget?", "text": "The budget is $100.",
         "cited_chunk_ids": ["stable-a"]}]}


@pytest.fixture
def audit():
    return {"evaluations": [{"claim_id": "claim-budget", "verdict": "supported",
        "evidence": [{"chunk_id": "stable-a", "excerpt": "The budget is $100."}],
        "conflicting_evidence": [], "explanation": "The cited text states the same budget."}],
        "part_evaluations": [{"part": "What is the budget?", "is_answered": True, "explanation": "Budget given."}],
        "unanswered_parts": []}


def run_mocked(monkeypatch, evidence, draft, audit, relevant=None, query=None):
    """Keep generation/audit provider boundaries real, mock their returned JSON only."""
    prompts = []
    monkeypatch.setattr(main, "ingest_document", lambda *a: SimpleNamespace(
        file_name="budget.pdf", id="doc-a", file_hash="content-hash"))
    monkeypatch.setattr(main, "chroma_vector_retrieval", lambda *a: evidence)
    def provider(prompt, schema, **kwargs):
        prompts.append((schema, prompt))
        if schema is BatchContextVerification:
            result = {"evaluations": [{"chunk_id": c.chunk_id, "is_relevant": relevant is None or c.chunk_id in relevant,
                                       "reason": "mock relevance"} for c in evidence]}
        elif schema is StructuredAnswer:
            result = draft
        elif schema is ClaimAudit:
            result = audit
        else:
            raise AssertionError(f"Unexpected schema: {schema}")
        return json.dumps(result)
    monkeypatch.setattr(config, "generate_structured_json", provider)
    result = main.run_pipeline("unused.pdf", query or " ".join(draft["requested_parts"]))
    return result, prompts


def test_valid_support_renders_exact_claim_and_source(monkeypatch, evidence, draft, audit):
    result, prompts = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "answered"
    assert result.answer == "The budget is $100. [budget.pdf, p. 2; stable-a]"
    claim = result.claims[0]
    assert claim.verdict == claim.judgment.verdict == "supported"
    assert claim.checks.passed
    assert claim.evidence[0].excerpt == "The budget is $100."
    assert claim.evidence[0].file_name == "budget.pdf"
    assert claim.evidence[0].page_number == 2
    assert not result.unanswered_parts
    assert result.sources == evidence[:1]
    # All model stages explicitly distrust source content; draft/audit add no extra loop.
    assert [s for s, _ in prompts] == [BatchContextVerification, StructuredAnswer, ClaimAudit]
    assert all("untrusted data" in prompt for _, prompt in prompts)


def test_fabricated_claim_citation_overrides_supported_model(monkeypatch, evidence, draft, audit):
    draft["claims"][0]["cited_chunk_ids"] = ["invented-id"]
    result, _ = run_mocked(monkeypatch, evidence, draft, audit)
    assert result.outcome == "abstained" and result.answer is None
    assert result.claims[0].judgment.verdict == "supported"
    assert result.claims[0].verdict == "insufficient_evidence"
    assert not result.claims[0].checks.passed
    assert "Cited IDs are outside retrieved evidence" in result.claims[0].checks.errors
    assert result.unanswered_parts == draft["requested_parts"]


@pytest.mark.parametrize("quote", [
    {"chunk_id": "invented", "excerpt": "The budget is $100."},
    {"chunk_id": "stable-a", "excerpt": "The budget is $200."},  # Exists only in the other chunk.
    {"chunk_id": "stable-b", "excerpt": "The budget is $200."},  # Real, but not cited by the claim.
    {"chunk_id": "stable-a", "excerpt": "   "},
])
def test_fabricated_or_miscited_excerpt_is_blocked(monkeypatch, evidence, draft, audit, quote):
    audit["evaluations"][0]["evidence"] = [quote]
    result, _ = run_mocked(monkeypatch, evidence, draft, audit)
    assert result.outcome == "abstained" and result.answer is None
    assert not result.claims[0].checks.passed


def test_unsupported_number_requires_model_judgment(monkeypatch, evidence, draft, audit):
    draft["claims"][0]["text"] = "The budget grows 9% each year."
    audit["evaluations"][0].update(verdict="insufficient_evidence", explanation="The excerpt gives no annual growth rate.")
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.claims[0].checks.passed  # Correct provenance alone cannot establish semantic support.
    assert result.claims[0].verdict == "insufficient_evidence"
    assert result.answer is None and result.outcome == "abstained"
    assert result.unanswered_parts == draft["requested_parts"]
    assert result.claims[0].evidence[0].excerpt == "The budget is $100."


def test_contradicted_claim_is_not_rendered(monkeypatch, evidence, draft, audit):
    draft["claims"][0]["text"] = "The budget is $200."
    audit["evaluations"][0].update(verdict="contradicted", explanation="This source states $100, not $200.")
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.claims[0].checks.passed
    assert result.claims[0].verdict == "contradicted"
    assert result.answer is None and result.outcome == "abstained"


def test_source_disagreement_survives_relevance_filter(monkeypatch, evidence, draft, audit):
    audit["evaluations"][0]["conflicting_evidence"] = [{"chunk_id": "stable-b", "excerpt": "The budget is $200."}]
    audit["evaluations"][0]["explanation"] = "The two sources give different budgets."
    result, prompts = run_mocked(monkeypatch, evidence, draft, audit, relevant={"stable-a"})
    generation_prompt = next(prompt for schema, prompt in prompts if schema is StructuredAnswer)
    auditor_prompt = next(prompt for schema, prompt in prompts if schema is ClaimAudit)
    assert "stable-b" not in generation_prompt and "stable-b" in auditor_prompt
    # Conservative gate does not allow a supported label to hide an explicit conflict.
    assert result.claims[0].judgment.verdict == "supported"
    assert result.claims[0].verdict == "contradicted"
    assert result.answer is None and result.outcome == "abstained"
    assert len(result.disagreements) == 1
    refs = result.disagreements[0].evidence
    assert {(q.file_name, q.page_number) for q in refs} == {("budget.pdf", 2), ("revision.pdf", 5)}
    assert {q.chunk_id for q in refs} == {"stable-a", "stable-b"}
    assert {c.chunk_id for c in result.sources} == {"stable-a", "stable-b"}


def test_fabricated_conflict_not_surfaced(monkeypatch, evidence, draft, audit):
    audit["evaluations"][0]["conflicting_evidence"] = [{"chunk_id": "stable-b", "excerpt": "There is no budget."}]
    result, _ = run_mocked(monkeypatch, evidence, draft, audit)
    assert result.answer is None
    assert not result.claims[0].checks.passed
    assert not result.disagreements


def add_launch_claim(draft, audit, verdict="insufficient_evidence"):
    draft["requested_parts"].append("When is the launch?")
    draft["claims"].append({"claim_id": "claim-launch", "part": "When is the launch?",
                            "text": "The launch is in July.", "cited_chunk_ids": ["stable-a"]})
    audit["evaluations"].append({"claim_id": "claim-launch", "verdict": verdict,
        "evidence": [{"chunk_id": "stable-a", "excerpt": "The launch is in June."}],
        "conflicting_evidence": [], "explanation": "The evidence does not support July."})
    audit["part_evaluations"].append({"part": "When is the launch?", "is_answered": True, "explanation": "mock overconfidence"})


def test_partial_answer_cannot_hide_dropped_claim(monkeypatch, evidence, draft, audit):
    add_launch_claim(draft, audit)
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "partial"
    assert result.answer == "The budget is $100. [budget.pdf, p. 2; stable-a]"
    assert "July" not in result.answer
    assert result.unanswered_parts == ["When is the launch?"]
    assert len(result.claims) == 2


def test_dropped_claim_keeps_same_question_part_unresolved(monkeypatch, evidence, draft, audit):
    add_launch_claim(draft, audit)
    draft["claims"][1]["part"] = "What is the budget?"
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "partial"
    assert set(result.unanswered_parts) == {"What is the budget?", "When is the launch?"}


def test_part_without_generated_claim_remains_unanswered(monkeypatch, evidence, draft, audit):
    draft["requested_parts"].append("Who approved it?")
    audit["part_evaluations"].append({"part": "Who approved it?", "is_answered": True, "explanation": "mock overconfidence"})
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "partial" and result.unanswered_parts == ["Who approved it?"]


@pytest.mark.parametrize("origin", ["generator", "auditor", "coverage"])
def test_model_reported_gaps_preserved(monkeypatch, evidence, draft, audit, origin):
    if origin == "generator":
        draft["unanswered_parts"] = ["Who approved it?"]
    elif origin == "auditor":
        audit["unanswered_parts"] = ["Who approved it?"]  # The generator omitted this question part entirely.
    else:
        audit["part_evaluations"][0]["is_answered"] = False
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "partial" and result.unanswered_parts


@pytest.mark.parametrize("bad_ids", [[], ["invented"], ["claim-budget", "claim-budget"]])
def test_missing_duplicate_unknown_claim_evaluations_fail_closed(monkeypatch, evidence, draft, audit, bad_ids):
    judgment = audit["evaluations"][0]
    audit["evaluations"] = [{**judgment, "claim_id": identity} for identity in bad_ids]
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "abstained" and result.answer is None
    assert result.error_category == "malformed_output"
    assert result.unanswered_parts


def test_missing_part_audit_fails_closed(monkeypatch, evidence, draft, audit):
    audit["part_evaluations"] = []
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "abstained" and result.error_category == "malformed_output"


def test_no_claims_abstains(monkeypatch, evidence, draft, audit):
    draft["claims"] = []
    audit["evaluations"] = []
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "abstained" and result.answer is None
    assert result.unanswered_parts == draft["requested_parts"]


def test_duplicate_claim_ids_fail_before_audit(monkeypatch, evidence, draft, audit):
    draft["claims"].append(copy.deepcopy(draft["claims"][0]))
    result, prompts = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "abstained" and result.error_category == "malformed_output"
    assert len(prompts) == 2


def test_no_page_reference_is_invented(monkeypatch, evidence, draft, audit):
    evidence[0].page_number = None
    evidence[0].file_name = "budget.txt"
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.answer == "The budget is $100. [budget.txt; stable-a]"
    assert result.claims[0].evidence[0].page_number is None


@pytest.mark.parametrize("problem", ["duplicate_citations", "missing_excerpt"])
def test_missing_or_duplicate_provenance_blocks_claim(monkeypatch, evidence, draft, audit, problem):
    if problem == "duplicate_citations":
        draft["claims"][0]["cited_chunk_ids"] *= 2
    else:
        audit["evaluations"][0]["evidence"] = []
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert not result.claims[0].checks.passed
    assert result.outcome == "abstained" and result.answer is None


def test_bad_citation_drops_only_its_claim_and_marks_partial(monkeypatch, evidence, draft, audit):
    add_launch_claim(draft, audit, verdict="supported")
    draft["claims"][1]["cited_chunk_ids"] = ["invented"]
    result, _ = run_mocked(monkeypatch, evidence[:1], draft, audit)
    assert result.outcome == "partial" and result.answer.startswith("The budget is $100.")
    assert "July" not in result.answer
    assert result.unanswered_parts == ["When is the launch?"]
    assert result.claims[0].checks.passed and not result.claims[1].checks.passed
