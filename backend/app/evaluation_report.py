"""Combined evaluation report bringing together retrieval and answer quality metrics."""

from typing import List, Optional

from pydantic import BaseModel, Field

try:
    from backend.app.answer_quality_evaluator import (
        AnswerQualityEvaluator,
        AnswerQualityScore,
    )
    from backend.app.evaluation_runner import EvaluationResult, EvaluationRunResult
    from backend.app.retrieval_evaluator import (
        RetrievalEvaluation,
        RetrievalEvaluator,
    )
except ImportError:
    from app.answer_quality_evaluator import (
        AnswerQualityEvaluator,
        AnswerQualityScore,
    )
    from app.evaluation_runner import EvaluationResult, EvaluationRunResult
    from app.retrieval_evaluator import (
        RetrievalEvaluation,
        RetrievalEvaluator,
    )


class AnswerQualityAggregate(BaseModel):
    """Aggregated answer quality scores across evaluated cases."""

    average_correctness: float = Field(
        ..., ge=0.0, le=1.0, description="Mean correctness score across all cases."
    )
    average_groundedness: float = Field(
        ..., ge=0.0, le=1.0, description="Mean groundedness score across all cases."
    )
    average_completeness: float = Field(
        ..., ge=0.0, le=1.0, description="Mean completeness score across all cases."
    )
    average_overall: float = Field(
        ..., ge=0.0, le=1.0, description="Mean overall quality score across all cases."
    )
    total_cases: int = Field(
        ..., ge=0, description="Total number of evaluated cases."
    )


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report containing retrieval and generation quality metrics."""

    retrieval: RetrievalEvaluation
    answer_quality: AnswerQualityAggregate
    total_cases: int = Field(..., ge=0, description="Total cases evaluated.")


class EvaluationReportBuilder:
    """Composes retrieval evaluation and answer quality into a unified report."""

    def __init__(
        self,
        retrieval_evaluator: Optional[RetrievalEvaluator] = None,
        answer_quality_evaluator: Optional[AnswerQualityEvaluator] = None,
    ) -> None:
        """Initialize the report builder with evaluators.

        Args:
            retrieval_evaluator: Optional RetrievalEvaluator instance.
            answer_quality_evaluator: Optional AnswerQualityEvaluator instance.
        """
        self.retrieval_evaluator = retrieval_evaluator or RetrievalEvaluator()
        self.answer_quality_evaluator = answer_quality_evaluator or AnswerQualityEvaluator()

    def build(
        self,
        run_result: EvaluationRunResult,
        answer_quality_scores: List[AnswerQualityScore],
        retrieval_k: int,
    ) -> EvaluationReport:
        """Build a combined EvaluationReport from run results and answer quality scores.

        Args:
            run_result: The executed benchmark run results.
            answer_quality_scores: Already-computed AnswerQualityScore for each case.
            retrieval_k: Top-k parameter cutoff for retrieval evaluation.

        Returns:
            Unified EvaluationReport with retrieval and answer quality summaries.

        Raises:
            ValueError: If inputs are invalid or score count mismatches run result cases.
        """
        if run_result is None or not isinstance(run_result, EvaluationRunResult):
            raise ValueError("run_result must be a valid EvaluationRunResult instance.")

        if not isinstance(answer_quality_scores, list):
            raise ValueError("answer_quality_scores must be a list of AnswerQualityScore.")

        total_cases = len(run_result.results)

        if len(answer_quality_scores) != total_cases:
            raise ValueError(
                f"Mismatch between number of answer quality scores ({len(answer_quality_scores)}) "
                f"and evaluation results ({total_cases})."
            )

        # Delegate retrieval metric calculation to the existing RetrievalEvaluator
        retrieval_evaluation = self.retrieval_evaluator.evaluate(
            run_result=run_result,
            k=retrieval_k,
        )

        if total_cases == 0:
            answer_quality_aggregate = AnswerQualityAggregate(
                average_correctness=0.0,
                average_groundedness=0.0,
                average_completeness=0.0,
                average_overall=0.0,
                total_cases=0,
            )
        else:
            avg_correctness = (
                sum(score.correctness for score in answer_quality_scores) / total_cases
            )
            avg_groundedness = (
                sum(score.groundedness for score in answer_quality_scores) / total_cases
            )
            avg_completeness = (
                sum(score.completeness for score in answer_quality_scores) / total_cases
            )
            avg_overall = (
                sum(score.overall for score in answer_quality_scores) / total_cases
            )

            answer_quality_aggregate = AnswerQualityAggregate(
                average_correctness=float(avg_correctness),
                average_groundedness=float(avg_groundedness),
                average_completeness=float(avg_completeness),
                average_overall=float(avg_overall),
                total_cases=total_cases,
            )

        return EvaluationReport(
            retrieval=retrieval_evaluation,
            answer_quality=answer_quality_aggregate,
            total_cases=total_cases,
        )

    def build_from_results(
        self,
        run_result: EvaluationRunResult,
        retrieval_k: int,
        answer_quality_evaluator: Optional[AnswerQualityEvaluator] = None,
        contexts_list: Optional[List[List[str]]] = None,
    ) -> EvaluationReport:
        """Convenience method evaluating answer quality directly before building report.

        Args:
            run_result: The executed benchmark run results.
            retrieval_k: Top-k parameter cutoff for retrieval evaluation.
            answer_quality_evaluator: Optional evaluator override.
            contexts_list: Optional list of retrieved text contexts per case.

        Returns:
            Unified EvaluationReport.
        """
        evaluator = answer_quality_evaluator or self.answer_quality_evaluator
        scores: List[AnswerQualityScore] = []

        for index, result in enumerate(run_result.results):
            contexts = contexts_list[index] if contexts_list and index < len(contexts_list) else []
            scores.append(
                evaluator.evaluate_case(
                    question=result.question,
                    expected_answer=result.expected_answer,
                    actual_answer=result.actual_answer,
                    context=contexts,
                )
            )

        return self.build(
            run_result=run_result,
            answer_quality_scores=scores,
            retrieval_k=retrieval_k,
        )
