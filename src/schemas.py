from pydantic import BaseModel, Field
from typing import List

class ChunkEvaluation(BaseModel):
    chunk_index: int = Field(description="The matching index of the checked chunk.")
    is_relevant: bool = Field(description="True if this chunk directly helps answer the query.")
    reason: str = Field(description="Brief justification for the decision.")

class BatchContextVerification(BaseModel):
    evaluations: List[ChunkEvaluation] = Field(description="List of evaluations for every provided chunk.")

class HallucinationCheck(BaseModel):
    is_faithful: bool = Field(description="True if the response relies ONLY on the verified context.")
    verdict: str = Field(description="Explanation of alignment.")