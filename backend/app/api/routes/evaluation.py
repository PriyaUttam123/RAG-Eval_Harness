"""Evaluation API route module."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from backend.app.evaluation_service import EvaluationService
except ImportError:
    from app.evaluation_service import EvaluationService


router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


class EvaluationRunRequest(BaseModel):
    """Request payload for initiating an evaluation benchmark run."""

    dataset: str = Field(
        ..., description="Dataset to evaluate ('golden' or 'holdout')."
    )
    retrieval_k: int = Field(
        ..., gt=0, description="Top-k evaluation cutoff depth for retrieval metrics."
    )
    top_k: Optional[int] = Field(
        default=None, gt=0, description="Optional top-k chunks retrieved for each case."
    )


@router.post("/run")
def run_evaluation(request: EvaluationRunRequest):
    """Execute benchmark evaluation across a golden or holdout dataset."""
    try:
        service = EvaluationService()
        report = service.run_dataset(
            dataset_name=request.dataset,
            retrieval_k=request.retrieval_k,
            top_k=request.top_k,
        )
        return report.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
