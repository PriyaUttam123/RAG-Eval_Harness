"""Service for executing configured RAG experiments."""

from time import perf_counter
from typing import Any, Optional

try:
    from backend.app.evaluation_report import (
        AnswerQualityAggregate,
        EvaluationReport,
    )
    from backend.app.evaluation_service import EvaluationService
    from backend.app.experiment_config import ExperimentConfig
    from backend.app.experiment_models import ExperimentResult
    from backend.app.experiment_service import ExperimentService
    from backend.app.retrieval_evaluator import RetrievalEvaluation
except ImportError:
    from app.evaluation_report import (
        AnswerQualityAggregate,
        EvaluationReport,
    )
    from app.evaluation_service import EvaluationService
    from app.experiment_config import ExperimentConfig
    from app.experiment_models import ExperimentResult
    from app.experiment_service import ExperimentService
    from app.retrieval_evaluator import RetrievalEvaluation


class ExperimentExecutionService:
    """Execute configured RAG experiments."""

    def __init__(
        self,
        evaluation_service: Optional[EvaluationService] = None,
        experiment_service: Optional[ExperimentService] = None,
    ) -> None:
        self.evaluation_service = evaluation_service or EvaluationService()
        self.experiment_service = experiment_service or ExperimentService()

    def run(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        dataset_name: str = "golden",
    ) -> ExperimentResult:
        """Execute one experiment and return its result."""

        run_metadata = self.experiment_service.create_run_metadata(
            experiment_id=experiment_id,
            config=config,
            dataset_name=dataset_name,
        )

        started_at = perf_counter()

        try:
            report = self.evaluation_service.run_dataset(
                dataset_name=dataset_name,
                retrieval_k=config.evaluation_k,
                top_k=config.retrieval_top_k,
            )

            duration_seconds = perf_counter() - started_at

            return ExperimentResult(
                run_metadata=run_metadata,
                report=report,
                duration_seconds=duration_seconds,
                error_count=0,
                metadata=self._build_metadata(config),
            )

        except Exception:
            duration_seconds = perf_counter() - started_at

            return ExperimentResult(
                run_metadata=run_metadata,
                report=self._empty_report(),
                duration_seconds=duration_seconds,
                error_count=1,
                metadata=self._build_metadata(config),
            )

    @staticmethod
    def _build_metadata(config: ExperimentConfig) -> dict[str, Any]:
        """Build metadata describing the executed experiment."""

        return {
            "retrieval_strategy": config.retrieval_strategy,
            "reranking_enabled": config.reranking_enabled,
            "embedding_model": config.embedding_model,
            "generation_model": config.generation_model,
        }

    @staticmethod
    def _empty_report() -> EvaluationReport:
        """Create a zero-case report when execution fails."""

        return EvaluationReport(
            retrieval=RetrievalEvaluation(
                k=1,
                hit_rate=0.0,
                recall=0.0,
                mrr=0.0,
                total_cases=0,
            ),
            answer_quality=AnswerQualityAggregate(
                average_correctness=0.0,
                average_groundedness=0.0,
                average_completeness=0.0,
                average_overall=0.0,
                total_cases=0,
            ),
            total_cases=0,
        )
