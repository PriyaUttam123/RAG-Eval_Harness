
"""Combined evaluation report bringing together metrics and failure analysis."""

from typing import List, Optional

from pydantic import BaseModel, Field

try:
    from backend.app.answer_quality_evaluator import (
        AnswerQualityEvaluator,
        AnswerQualityScore,
    )
    from backend.app.evaluation_runner import EvaluationResult, EvaluationRunResult
    from backend.app.failure_analysis import EvaluationFailure, FailureAnalysis
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
    from app.failure_analysis import EvaluationFailure, FailureAnalysis
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
    """Comprehensive evaluation report containing metrics and failure details."""

    retrieval: RetrievalEvaluation
    answer_quality: AnswerQualityAggregate
    failure_analysis: FailureAnalysis
    total_cases: int = Field(..., ge=0, description="Total cases evaluated.")


class EvaluationReportBuilder:
    """Composes retrieval metrics, answer quality, and failure analysis."""

    def __init__(
        self,
        retrieval_evaluator: Optional[RetrievalEvaluator] = None,
        answer_quality_evaluator: Optional[AnswerQualityEvaluator] = None,
    ) -> None:
        """Initialize the report builder with evaluators."""
        self.retrieval_evaluator = retrieval_evaluator or RetrievalEvaluator()
        self.answer_quality_evaluator = (
            answer_quality_evaluator or AnswerQualityEvaluator()
        )

    def build(
        self,
        run_result: EvaluationRunResult,
        answer_quality_scores: List[AnswerQualityScore],
        retrieval_k: int,
    ) -> EvaluationReport:
        """Build a report from evaluation results and answer quality scores."""
        if run_result is None or not isinstance(run_result, EvaluationRunResult):
            raise ValueError(
                "run_result must be a valid EvaluationRunResult instance."
            )

        if not isinstance(answer_quality_scores, list):
            raise ValueError(
                "answer_quality_scores must be a list of AnswerQualityScore."
            )

        total_cases = len(run_result.results)

        if len(answer_quality_scores) != total_cases:
            raise ValueError(
                f"Mismatch between number of answer quality scores "
                f"({len(answer_quality_scores)}) and evaluation results "
                f"({total_cases})."
            )

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
            answer_quality_aggregate = AnswerQualityAggregate(
                average_correctness=float(
                    sum(score.correctness for score in answer_quality_scores)
                    / total_cases
                ),
                average_groundedness=float(
                    sum(score.groundedness for score in answer_quality_scores)
                    / total_cases
                ),
                average_completeness=float(
                    sum(score.completeness for score in answer_quality_scores)
                    / total_cases
                ),
                average_overall=float(
                    sum(score.overall for score in answer_quality_scores)
                    / total_cases
                ),
                total_cases=total_cases,
            )

        # Identify execution, retrieval, and low answer-quality failures.
        failures: List[EvaluationFailure] = []

        for index, (result, score) in enumerate(
            zip(run_result.results, answer_quality_scores), start=1
        ):
            failure_type = None

            if result.error:
                failure_type = "execution_error"
            elif (
                result.relevant_chunk_ids
                and not set(result.retrieved_chunk_ids).intersection(
                    result.relevant_chunk_ids
                )
            ):
                failure_type = "retrieval_failure"
            elif score.overall < 0.5:
                failure_type = "answer_quality"

            if failure_type is not None:
                failures.append(
                    EvaluationFailure(
                        case_id=f"case-{index:03d}",
                        question=result.question,
                        expected_answer=result.expected_answer,
                        actual_answer=result.actual_answer,
                        failure_type=failure_type,
                        error_message=result.error,
                        retrieved_chunk_ids=result.retrieved_chunk_ids,
                        answer_quality_score=score.overall,
                    )
                )

        failure_analysis = FailureAnalysis(
            failures=failures,
            total_failures=len(failures),
        )

        return EvaluationReport(
            retrieval=retrieval_evaluation,
            answer_quality=answer_quality_aggregate,
            failure_analysis=failure_analysis,
            total_cases=total_cases,
        )

    def build_from_results(
        self,
        run_result: EvaluationRunResult,
        retrieval_k: int,
        answer_quality_evaluator: Optional[AnswerQualityEvaluator] = None,
        contexts_list: Optional[List[List[str]]] = None,
    ) -> EvaluationReport:
        """Evaluate answer quality and build a combined report."""
        evaluator = answer_quality_evaluator or self.answer_quality_evaluator
        scores: List[AnswerQualityScore] = []

        for index, result in enumerate(run_result.results):
            contexts = (
                contexts_list[index]
                if contexts_list and index < len(contexts_list)
                else []
            )
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
