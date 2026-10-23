"""Service for executing configured RAG experiments."""

from time import perf_counter
from typing import Any, Optional

try:
    from backend.app.configured_retrieval_service import ConfiguredRetrievalService
    from backend.app.evaluation_report import AnswerQualityAggregate, EvaluationReport
    from backend.app.evaluation_runner import EvaluationRunner
    from backend.app.evaluation_service import EvaluationService
    from backend.app.experiment_config import ExperimentConfig
    from backend.app.experiment_models import ExperimentResult
    from backend.app.experiment_retrieval_service import ExperimentRetrievalService
    from backend.app.experiment_service import ExperimentService
    from backend.app.rag_service import RAGService
    from backend.app.retrieval_evaluator import RetrievalEvaluation
except ImportError:
    from app.configured_retrieval_service import ConfiguredRetrievalService
    from app.evaluation_report import AnswerQualityAggregate, EvaluationReport
    from app.evaluation_runner import EvaluationRunner
    from app.evaluation_service import EvaluationService
    from app.experiment_config import ExperimentConfig
    from app.experiment_models import ExperimentResult
    from app.experiment_retrieval_service import ExperimentRetrievalService
    from app.experiment_service import ExperimentService
    from app.rag_service import RAGService
    from app.retrieval_evaluator import RetrievalEvaluation


class ExperimentExecutionService:
    def __init__(
        self,
        evaluation_service=None,
        experiment_service=None,
        experiment_retrieval_service: Optional[
            ExperimentRetrievalService
        ] = None,
        generation_service=None,
    ):
        self.evaluation_service = evaluation_service
        self.experiment_service = experiment_service or ExperimentService()
        self.experiment_retrieval_service = experiment_retrieval_service
        self.generation_service = generation_service

    def run(
        self,
        experiment_id,
        config,
        dataset_name="golden",
    ):
        run_metadata = self.experiment_service.create_run_metadata(
            experiment_id=experiment_id,
            config=config,
            dataset_name=dataset_name,
        )

        started_at = perf_counter()

        try:
            evaluation_service = self._get_evaluation_service(config)

            report = evaluation_service.run_dataset(
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

    def _get_evaluation_service(self, config: ExperimentConfig):
        if self.evaluation_service is not None:
            return self.evaluation_service

        if self.experiment_retrieval_service is None:
            return EvaluationService()

        configured_retrieval = ConfiguredRetrievalService(
            experiment_retrieval_service=self.experiment_retrieval_service,
            config=config,
        )

        rag_service = RAGService(
            retrieval_service=configured_retrieval,
            generation_service=self.generation_service,
        )

        runner = EvaluationRunner(rag_service=rag_service)

        return EvaluationService(runner=runner)

    @staticmethod
    def _build_metadata(config):
        return {
            "retrieval_strategy": config.retrieval_strategy,
            "reranking_enabled": config.reranking_enabled,
            "embedding_model": config.embedding_model,
            "generation_model": config.generation_model,
        }

    @staticmethod
    def _empty_report():
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
