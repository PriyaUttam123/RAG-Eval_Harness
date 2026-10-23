"""Experiment comparison models and service."""

from typing import List

try:
    from backend.app.experiment_models import ExperimentResult
except ImportError:
    from app.experiment_models import ExperimentResult

from pydantic import BaseModel, Field


class ExperimentComparisonRow(BaseModel):
    """Comparable metrics for one experiment result."""

    experiment_id: str
    experiment_name: str

    hit_rate: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    mrr: float = Field(..., ge=0.0, le=1.0)

    average_correctness: float = Field(..., ge=0.0, le=1.0)
    average_groundedness: float = Field(..., ge=0.0, le=1.0)
    average_completeness: float = Field(..., ge=0.0, le=1.0)
    average_overall: float = Field(..., ge=0.0, le=1.0)

    duration_seconds: float | None = Field(default=None, ge=0.0)
    error_count: int = Field(default=0, ge=0)


class ExperimentComparison(BaseModel):
    """Comparison of multiple executed experiments."""

    rows: List[ExperimentComparisonRow]
    total_experiments: int = Field(..., ge=0)


class ExperimentComparisonService:
    """Build a comparable view from executed experiment results."""

    def compare(
        self,
        results: List[ExperimentResult],
    ) -> ExperimentComparison:
        """Compare executed experiment results without inventing a winner score."""

        rows = [
            ExperimentComparisonRow(
                experiment_id=result.run_metadata.experiment_id,
                experiment_name=result.run_metadata.config.name,
                hit_rate=result.report.retrieval.hit_rate,
                recall=result.report.retrieval.recall,
                mrr=result.report.retrieval.mrr,
                average_correctness=result.report.answer_quality.average_correctness,
                average_groundedness=result.report.answer_quality.average_groundedness,
                average_completeness=result.report.answer_quality.average_completeness,
                average_overall=result.report.answer_quality.average_overall,
                duration_seconds=result.duration_seconds,
                error_count=result.error_count,
            )
            for result in results
        ]

        return ExperimentComparison(
            rows=rows,
            total_experiments=len(rows),
        )
