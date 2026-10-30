
"""Evaluation API route module."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import AliasChoices, BaseModel, Field

try:
    from backend.app.evaluation_service import EvaluationService
    from backend.app.experiment_comparison import ExperimentComparisonService
    from backend.app.experiment_config import ExperimentConfig
    from backend.app.experiment_execution_service import (
        ExperimentExecutionService,
    )
    from backend.app.experiment_models import ExperimentResult
except ImportError:
    from app.evaluation_service import EvaluationService
    from app.experiment_comparison import ExperimentComparisonService
    from app.experiment_config import ExperimentConfig
    from app.experiment_execution_service import (
        ExperimentExecutionService,
    )
    from app.experiment_models import ExperimentResult


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


class SingleExperimentRunRequest(BaseModel):
    """Request payload for executing a single configured experiment."""

    experiment_id: str = Field(
        ..., description="Unique identifier for the experiment run."
    )
    config: ExperimentConfig = Field(
        ..., description="The experiment configuration."
    )
    dataset_name: str = Field(
        default="golden",
        validation_alias=AliasChoices("dataset_name", "dataset"),
        description="Dataset to evaluate ('golden' or 'holdout').",
    )


@router.post("/experiments/run")
def run_single_experiment(request: SingleExperimentRunRequest):
    """Execute a single configured RAG experiment against a dataset."""
    try:
        service = ExperimentExecutionService()
        result = service.run(
            experiment_id=request.experiment_id,
            config=request.config,
            dataset_name=request.dataset_name,
        )
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during experiment execution.",
        ) from exc


class BatchExperimentItem(BaseModel):
    """Single experiment item in a batch execution request."""

    experiment_id: str = Field(
        ..., description="Unique identifier for the experiment run."
    )
    config: ExperimentConfig = Field(
        ..., description="The experiment configuration."
    )


class BatchExperimentRunRequest(BaseModel):
    """Request payload for executing a batch of configured experiments."""

    experiments: List[BatchExperimentItem] = Field(
        ..., description="Ordered collection of experiment configurations."
    )
    dataset_name: str = Field(
        default="golden",
        validation_alias=AliasChoices("dataset_name", "dataset"),
        description="Dataset to evaluate ('golden' or 'holdout').",
    )


@router.post("/experiments/run-batch")
def run_batch_experiments(request: BatchExperimentRunRequest):
    """Execute multiple configured RAG experiments in order against a dataset."""
    try:
        service = ExperimentExecutionService()
        experiment_tuples = [
            (item.experiment_id, item.config)
            for item in request.experiments
        ]
        results = service.run_many(
            experiments=experiment_tuples,
            dataset_name=request.dataset_name,
        )
        return [result.model_dump() for result in results]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during batch experiment execution.",
        ) from exc
