from pydantic import BaseModel, Field

class ContextVerification(BaseModel):
    is_relevant: bool = Field(description="True if the chunk directly helps answer the user query.")
    reason: str = Field(description="Brief justification for the decision.")

class HallucinationCheck(BaseModel):
    is_faithful: bool = Field(description="True if the response relies ONLY on the verified context without hallucinating.")
    verdict: str = Field(description="Explanation of any fabrications, or confirmation of alignment.")

