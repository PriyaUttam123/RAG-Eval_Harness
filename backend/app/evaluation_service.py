"""End-to-end evaluation orchestration service."""

from typing import List, Optional

try:
    from backend.app.answer_quality_evaluator import (
        AnswerQualityEvaluator,
        AnswerQualityScore,
    )
    from backend.app.evaluation_dataset_manager import EvaluationDatasetManager
    from backend.app.evaluation_report import (
        EvaluationReport,
        EvaluationReportBuilder,
    )
    from backend.app.evaluation_runner import EvaluationResult, EvaluationRunner
    from backend.app.retrieval_evaluator import RetrievalEvaluator
except ImportError:
    from app.answer_quality_evaluator import (
        AnswerQualityEvaluator,
        AnswerQualityScore,
    )
    from app.evaluation_dataset_manager import EvaluationDatasetManager
    from app.evaluation_report import (
        EvaluationReport,
        EvaluationReportBuilder,
    )
    from app.evaluation_runner import EvaluationResult, EvaluationRunner
    from app.retrieval_evaluator import RetrievalEvaluator


class EvaluationService:
    """Orchestrates end-to-end evaluation runs across golden and holdout datasets."""

    def __init__(
        self,
        dataset_manager: Optional[EvaluationDatasetManager] = None,
        runner: Optional[EvaluationRunner] = None,
        retrieval_evaluator: Optional[RetrievalEvaluator] = None,
        answer_quality_evaluator: Optional[AnswerQualityEvaluator] = None,
        report_builder: Optional[EvaluationReportBuilder] = None,
    ) -> None:
        """Initialize EvaluationService with optional injected dependencies.

        Args:
            dataset_manager: Optional EvaluationDatasetManager instance.
            runner: Optional EvaluationRunner instance.
            retrieval_evaluator: Optional RetrievalEvaluator instance.
            answer_quality_evaluator: Optional AnswerQualityEvaluator instance.
            report_builder: Optional EvaluationReportBuilder instance.
        """
        self.dataset_manager = dataset_manager or EvaluationDatasetManager()
        self.runner = runner or EvaluationRunner()
        self.retrieval_evaluator = retrieval_evaluator or RetrievalEvaluator()
        self.answer_quality_evaluator = (
            answer_quality_evaluator or AnswerQualityEvaluator()
        )
        self.report_builder = report_builder or EvaluationReportBuilder(
            retrieval_evaluator=self.retrieval_evaluator,
            answer_quality_evaluator=self.answer_quality_evaluator,
        )

    def run_dataset(
        self,
        dataset_name: str,
        retrieval_k: int,
        top_k: Optional[int] = None,
    ) -> EvaluationReport:
        """Execute evaluation run for the specified dataset and return unified report.

        Args:
            dataset_name: Name of the dataset to evaluate ('golden' or 'holdout').
            retrieval_k: Cutoff depth for retrieval metrics evaluation.
            top_k: Optional number of chunks to retrieve during RAG generation.

        Returns:
            EvaluationReport containing retrieval metrics and answer quality aggregates.

        Raises:
            ValueError: If dataset_name is invalid or retrieval_k is non-positive.
        """
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError(
                "dataset_name must be a non-empty string ('golden' or 'holdout')."
            )

        normalized_name = dataset_name.strip().lower()
        if normalized_name == "golden":
            dataset = self.dataset_manager.load_golden()
        elif normalized_name == "holdout":
            dataset = self.dataset_manager.load_holdout()
        else:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. Supported datasets are 'golden' and 'holdout'."
            )

        # Execute evaluation cases through the runner
        run_result = self.runner.run(dataset=dataset, top_k=top_k)

        # Score answer quality for each individual case
        answer_quality_scores: List[AnswerQualityScore] = []
        for result in run_result.results:
            contexts = getattr(result, "retrieved_contexts", None)
            if contexts is None:
                contexts = getattr(result, "context", [])

            score = self.answer_quality_evaluator.evaluate_case(
                question=result.question,
                expected_answer=result.expected_answer,
                actual_answer=result.actual_answer,
                context=contexts or [],
            )
            answer_quality_scores.append(score)

        # Build and return the combined report
        return self.report_builder.build(
            run_result=run_result,
            answer_quality_scores=answer_quality_scores,
            retrieval_k=retrieval_k,
        )
