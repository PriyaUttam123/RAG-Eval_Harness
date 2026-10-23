
"""Evaluation API route module."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from backend.app.evaluation_service import EvaluationService
    from backend.app.experiment_comparison import ExperimentComparisonService
    from backend.app.experiment_models import ExperimentResult
except ImportError:
    from app.evaluation_service import EvaluationService
    from app.experiment_comparison import ExperimentComparisonService
    from app.experiment_models import ExperimentResult
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

class EvaluationComparisonRequest(BaseModel):
    """Request payload for comparing executed experiment results."""

    results: List[ExperimentResult] = Field(
        ...,
        description="Executed experiment results to compare.",
    )


@router.post("/compare")
def compare_evaluations(request: EvaluationComparisonRequest):
    """Compare multiple executed experiment results."""
    try:
        service = ExperimentComparisonService()
        comparison = service.compare(request.results)
        return comparison.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
