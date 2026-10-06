from uuid import UUID

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Validate the contract text submitted for analysis."""

    contract_text: str  = Field(..., min_length=1)


class AnalysisResponse(BaseModel):
    """Represent the result returned after starting an analysis."""

    run_id: UUID
    status: str
    report: str


class RunSummary(BaseModel):
    """Represent a saved analysis run in the run history."""

    run_id: UUID
    status: str
    contract_text: str


class EvalRunResponse(BaseModel):
    """Represent the latest persisted evaluation metrics."""
    
    model: str
    retrieval_enabled: bool
    accuracy: float
    mrr: float
    ndcg: float
    commit_hash: str