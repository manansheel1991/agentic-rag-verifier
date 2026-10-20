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
    outcome: Literal["success", "failure"]
    answer: Optional[str] = None
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
