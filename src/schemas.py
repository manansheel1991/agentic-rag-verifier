"""Typed records shared by ingestion, retrieval, critics and callers."""
import hashlib
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChunkEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: str
    is_relevant: bool
    reason: str


class BatchContextVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evaluations: List[ChunkEvaluation]


class HallucinationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    is_faithful: bool
    verdict: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    claim_id: str = Field(min_length=1)
    part: str = Field(min_length=1, description="Exact requested_parts entry this claim addresses.")
    text: str = Field(min_length=1, description="One atomic factual claim, including units and qualifications.")
    cited_chunk_ids: List[str] = Field(description="Cite retrieved evidence; an empty list is retained for audit, never assumed supported.")


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    requested_parts: List[str] = Field(min_length=1, description="Every part of the user's question, including unanswerable parts.")
    claims: List[Claim]
    unanswered_parts: List[str]


class EvidenceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: str
    excerpt: str = Field(min_length=1, description="Exact nonempty substring of this chunk; no paraphrasing or ellipses.")


class ClaimJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    claim_id: str
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    evidence: List[EvidenceQuote]
    conflicting_evidence: List[EvidenceQuote]
    explanation: str = Field(min_length=1)


class PartEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    part: str
    is_answered: bool
    explanation: str


class ClaimAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evaluations: List[ClaimJudgment]
    part_evaluations: List[PartEvaluation]
    unanswered_parts: List[str] = Field(description="Also include question parts omitted by the draft's requested_parts.")


class CitationChecks(BaseModel):
    passed: bool
    errors: List[str]


class SourceExcerpt(EvidenceQuote):
    document_id: str
    file_name: str
    page_number: Optional[int]


class VerifiedClaim(BaseModel):
    claim: Claim
    judgment: ClaimJudgment  # The model's semantic opinion, kept separate from Python checks.
    checks: CitationChecks
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    evidence: List[SourceExcerpt]
    conflicting_evidence: List[SourceExcerpt]


class SourceDisagreement(BaseModel):
    claim_id: str
    evidence: List[SourceExcerpt]
    explanation: str


class DocumentMetadata(BaseModel):
    id: str  # Canonical source path identity, independent of content/version/model.
    file_path: str
    file_hash: str
    last_modified: float
    file_name: str
    file_size: int
    embedding_model: str
    embedding_provider: str
    dimensions: int
    chunk_size: int
    chunk_overlap: int
    collection_name: str
    source_url: Optional[str] = None
    page_count: Optional[int] = None

    def get_hash(self):
        return hashlib.sha256(self.file_path.encode()).hexdigest()


class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    chunk_text_hash: str = ""
    text: str = ""
    file_hash: str = ""
    file_path: str = ""
    file_name: str = ""
    page_number: Optional[int] = None
    char_offset: int = Field(ge=0)
    char_length: int = Field(ge=0)

    def set_chunk_text(self, text):
        self.text = text
        self.chunk_text_hash = hashlib.sha256(text.encode()).hexdigest()

    def get_chunk_text(self):
        return self.text


class PipelineResult(BaseModel):
    outcome: Literal["answered", "partial", "abstained"]
    answer: Optional[str] = None
    claims: List[VerifiedClaim] = Field(default_factory=list)
    requested_parts: List[str] = Field(default_factory=list)
    unanswered_parts: List[str] = Field(default_factory=list)
    disagreements: List[SourceDisagreement] = Field(default_factory=list)
    sources: List[ChunkMetadata] = Field(default_factory=list)
    audit_log: List[str] = Field(default_factory=list)
    timings: Dict[str, float] = Field(default_factory=dict)
    embedding_model_used: str
    generation_model_used: Optional[str] = None
    model_identities: List[str] = Field(default_factory=list)
    tokens_used_embedding: Optional[int] = None
    tokens_used_generation: Optional[int] = None
    error_message: Optional[str] = None
    error_category: Optional[str] = None
