"""Deterministic citation gates and rendering; semantic judgments come from the model."""
from src.config import ProviderError
from src.schemas import (
    CitationChecks, ClaimAudit, PipelineResult, SourceDisagreement, SourceExcerpt,
    StructuredAnswer, VerifiedClaim,
)


def _require_exact(actual, expected, label):
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ProviderError("malformed_output", "claim_audit", [f"unknown, duplicate or missing {label}"])


def validate_draft(draft: StructuredAnswer):
    _require_exact(draft.requested_parts, draft.requested_parts, "question parts")
    _require_exact([c.claim_id for c in draft.claims], [c.claim_id for c in draft.claims], "claim IDs")
    if any(not part.strip() for part in draft.requested_parts + draft.unanswered_parts):
        raise ProviderError("malformed_output", "generator", ["empty question part"])
    if any(c.part not in draft.requested_parts or not c.text.strip() for c in draft.claims):
        raise ProviderError("malformed_output", "generator", ["invalid claim or question part"])


def verify_claims(draft: StructuredAnswer, audit: ClaimAudit, chunks) -> list[VerifiedClaim]:
    """Never infer semantic support from a valid citation or a matching substring."""
    validate_draft(draft)
    _require_exact([e.claim_id for e in audit.evaluations], [c.claim_id for c in draft.claims], "claim evaluations")
    _require_exact([p.part for p in audit.part_evaluations], draft.requested_parts, "part evaluations")
    if any(not part.strip() for part in audit.unanswered_parts):
        raise ProviderError("malformed_output", "claim_audit", ["empty unanswered part"])
    evidence = {chunk.chunk_id: chunk for chunk in chunks}
    if len(evidence) != len(chunks):
        raise ValueError("Duplicate retrieved evidence IDs")
    judgments = {e.claim_id: e for e in audit.evaluations}
    verified = []
    for claim in draft.claims:
        judgment = judgments[claim.claim_id]
        errors = []
        cited = set(claim.cited_chunk_ids)
        if not cited:
            errors.append("Missing chunk citations")
        if len(cited) != len(claim.cited_chunk_ids):
            errors.append("Duplicate cited chunk IDs")
        if cited - evidence.keys():
            errors.append("Cited IDs are outside retrieved evidence")
        if not judgment.evidence:
            errors.append("Missing source excerpt")
        if {quote.chunk_id for quote in judgment.evidence} != cited:
            errors.append("Evidence excerpts must cover exactly the claim's cited IDs")

        def resolve(quotes, must_be_cited):
            resolved = []
            for quote in quotes:
                chunk = evidence.get(quote.chunk_id)
                if chunk is None or must_be_cited and quote.chunk_id not in cited:
                    errors.append("Excerpt references an unknown or uncited chunk")
                elif not quote.excerpt.strip() or quote.excerpt not in chunk.text:
                    errors.append("Excerpt does not occur exactly in its chunk")
                else:
                    # Names and pages come from stored source records, never from the model.
                    resolved.append(SourceExcerpt(**quote.model_dump(), document_id=chunk.document_id,
                                                  file_name=chunk.file_name, page_number=chunk.page_number))
            return resolved

        sources = resolve(judgment.evidence, must_be_cited=True)
        conflicts = resolve(judgment.conflicting_evidence, must_be_cited=False)
        if conflicts and len({(q.chunk_id, q.excerpt) for q in sources + conflicts}) < 2:
            errors.append("Source disagreement requires distinct excerpts")
        verdict = judgment.verdict
        if errors:
            verdict = "insufficient_evidence"
        elif conflicts:
            # Conservatively withhold a disputed claim even if the model also called it supported.
            verdict = "contradicted"
        verified.append(VerifiedClaim(
            claim=claim, judgment=judgment, checks=CitationChecks(passed=not errors, errors=errors),
            verdict=verdict, evidence=sources, conflicting_evidence=conflicts))
    return verified


def render_claims(claims, sources):
    """Shared answer format; formatting alone makes no support judgment."""
    source_map = {source.chunk_id: source for source in sources}
    lines = []
    for claim in claims:
        references = []
        for identity in claim.cited_chunk_ids:
            source = source_map.get(identity)
            page = f", p. {source.page_number}" if source and source.page_number is not None else ""
            reference = f"[{source.file_name if source else 'unknown source'}{page}; {identity}]"
            if reference not in references:
                references.append(reference)
        lines.append(f"{claim.text} {' '.join(references)}".rstrip())
    return "\n".join(lines) or None


def render_supported_claims(claims: list[VerifiedClaim]):
    """No model-written preamble, conclusion, or other unaudited answer prose."""
    supported = [item for item in claims if item.verdict == "supported" and item.checks.passed]
    return render_claims([item.claim for item in supported], [q for item in supported for q in item.evidence])


def apply_claim_audit(result: PipelineResult, draft: StructuredAnswer, audit: ClaimAudit, chunks):
    result.claims = verify_claims(draft, audit, chunks)
    result.requested_parts = draft.requested_parts
    unanswered = [*draft.unanswered_parts, *audit.unanswered_parts]
    supported_parts = {item.claim.part for item in result.claims if item.verdict == "supported"}
    rejected_parts = {item.claim.part for item in result.claims if item.verdict != "supported"}
    coverage = {p.part: p.is_answered for p in audit.part_evaluations}
    for part in draft.requested_parts:
        # Any dropped claim keeps its part unresolved, even if another claim survived.
        if part not in supported_parts or part in rejected_parts or not coverage[part]:
            unanswered.append(part)
    result.unanswered_parts = list(dict.fromkeys(unanswered))
    result.answer = render_supported_claims(result.claims)
    result.outcome = "abstained" if result.answer is None else "partial" if unanswered else "answered"
    result.disagreements = [SourceDisagreement(
        claim_id=item.claim.claim_id, evidence=item.evidence + item.conflicting_evidence,
        explanation=item.judgment.explanation)
        for item in result.claims if item.checks.passed and item.conflicting_evidence]
    used_ids = {q.chunk_id for item in result.claims if item.verdict == "supported"
                for q in item.evidence}
    used_ids.update(q.chunk_id for d in result.disagreements for q in d.evidence)
    result.sources = [chunk for chunk in chunks if chunk.chunk_id in used_ids]
