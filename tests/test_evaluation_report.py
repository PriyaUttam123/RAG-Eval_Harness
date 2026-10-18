"""Unit and integration tests for EvaluationReportBuilder and EvaluationReport."""

import pytest
from pydantic import ValidationError

from backend.app.answer_quality_evaluator import AnswerQualityScore
from backend.app.evaluation_report import (
    AnswerQualityAggregate,
    EvaluationReport,
    EvaluationReportBuilder,
)
from backend.app.evaluation_runner import EvaluationResult, EvaluationRunResult
from backend.app.retrieval_evaluator import RetrievalEvaluation, RetrievalEvaluator


def _make_result(
    retrieved_chunk_ids,
    relevant_chunk_ids,
    question="Test question?",
    expected_answer="Expected answer.",
    actual_answer="Actual answer.",
    case_type="factual",
    error=None,
) -> EvaluationResult:
    return EvaluationResult(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        retrieved_chunk_ids=retrieved_chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
        case_type=case_type,
        error=error,
    )


def _make_run(results) -> EvaluationRunResult:
    return EvaluationRunResult(
        results=results,
        total_cases=len(results),
        successful_cases=sum(1 for r in results if r.error is None),
        failed_cases=sum(1 for r in results if r.error is not None),
    )


def test_perfect_combined_report():
    """Verify report values when both retrieval and answer quality are perfect."""
    run_result = _make_run(
        [
            _make_result(
                retrieved_chunk_ids=["chunk-1"],
                relevant_chunk_ids=["chunk-1"],
            ),
            _make_result(
                retrieved_chunk_ids=["chunk-2"],
                relevant_chunk_ids=["chunk-2"],
            ),
        ]
    )

    scores = [
        AnswerQualityScore(
            correctness=1.0,
            groundedness=1.0,
            completeness=1.0,
            overall=1.0,
            explanation="Perfect answer 1",
        ),
        AnswerQualityScore(
            correctness=1.0,
            groundedness=1.0,
            completeness=1.0,
            overall=1.0,
            explanation="Perfect answer 2",
        ),
    ]

    builder = EvaluationReportBuilder()
    report = builder.build(
        run_result=run_result,
        answer_quality_scores=scores,
        retrieval_k=1,
    )

    assert isinstance(report, EvaluationReport)
    assert report.total_cases == 2

    # Retrieval evaluation
    assert report.retrieval.hit_rate == pytest.approx(1.0)
    assert report.retrieval.recall == pytest.approx(1.0)
    assert report.retrieval.mrr == pytest.approx(1.0)
    assert report.retrieval.total_cases == 2

    # Answer quality aggregate
    assert report.answer_quality.average_correctness == pytest.approx(1.0)
    assert report.answer_quality.average_groundedness == pytest.approx(1.0)
    assert report.answer_quality.average_completeness == pytest.approx(1.0)
    assert report.answer_quality.average_overall == pytest.approx(1.0)
    assert report.answer_quality.total_cases == 2


def test_partial_retrieval_and_partial_answer_quality():
    """Verify accurate arithmetic averages with partial scores across metrics."""
    run_result = _make_run(
        [
            # Case 1: Hit=1.0, Recall=1.0, RR=1.0
            _make_result(
                retrieved_chunk_ids=["chunk-1"],
                relevant_chunk_ids=["chunk-1"],
            ),
            # Case 2: Hit=0.0, Recall=0.0, RR=0.0
            _make_result(
                retrieved_chunk_ids=["wrong-chunk"],
                relevant_chunk_ids=["chunk-2"],
            ),
        ]
    )

    scores = [
        AnswerQualityScore(
            correctness=0.8,
            groundedness=0.6,
            completeness=0.7,
            overall=0.7,
            explanation="Partial 1",
        ),
        AnswerQualityScore(
            correctness=0.4,
            groundedness=0.2,
            completeness=0.5,
            overall=0.3667,
            explanation="Partial 2",
        ),
    ]

    builder = EvaluationReportBuilder()
    report = builder.build(
        run_result=run_result,
        answer_quality_scores=scores,
        retrieval_k=1,
    )

    assert report.total_cases == 2

    # Retrieval: (1.0 + 0.0) / 2 = 0.5
    assert report.retrieval.hit_rate == pytest.approx(0.5)
    assert report.retrieval.recall == pytest.approx(0.5)
    assert report.retrieval.mrr == pytest.approx(0.5)

    # Answer quality
    assert report.answer_quality.average_correctness == pytest.approx((0.8 + 0.4) / 2)
    assert report.answer_quality.average_groundedness == pytest.approx((0.6 + 0.2) / 2)
    assert report.answer_quality.average_completeness == pytest.approx((0.7 + 0.5) / 2)
    assert report.answer_quality.average_overall == pytest.approx((0.7 + 0.3667) / 2)


def test_correct_averaging_across_multiple_cases():
    """Verify averaging precision with multiple varied evaluation cases."""
    results = [
        _make_result(["c1"], ["c1"]),
        _make_result(["c2"], ["c2"]),
        _make_result(["c3"], ["c3"]),
    ]
    run_result = _make_run(results)

    scores = [
        AnswerQualityScore(correctness=0.3, groundedness=0.6, completeness=0.9, overall=0.6),
        AnswerQualityScore(correctness=0.6, groundedness=0.6, completeness=0.6, overall=0.6),
        AnswerQualityScore(correctness=0.9, groundedness=0.6, completeness=0.3, overall=0.6),
    ]

    builder = EvaluationReportBuilder()
    report = builder.build(run_result, scores, retrieval_k=1)

    assert report.answer_quality.average_correctness == pytest.approx((0.3 + 0.6 + 0.9) / 3)
    assert report.answer_quality.average_groundedness == pytest.approx(0.6)
    assert report.answer_quality.average_completeness == pytest.approx((0.9 + 0.6 + 0.3) / 3)
    assert report.answer_quality.average_overall == pytest.approx(0.6)
    assert report.answer_quality.total_cases == 3


def test_empty_evaluation_run():
    """Verify an empty evaluation run produces zero-valued metrics and total_cases=0."""
    run_result = _make_run([])
    scores = []

    builder = EvaluationReportBuilder()
    report = builder.build(
        run_result=run_result,
        answer_quality_scores=scores,
        retrieval_k=3,
    )

    assert report.total_cases == 0
    assert report.retrieval.total_cases == 0
    assert report.retrieval.hit_rate == 0.0
    assert report.retrieval.recall == 0.0
    assert report.retrieval.mrr == 0.0

    assert report.answer_quality.total_cases == 0
    assert report.answer_quality.average_correctness == 0.0
    assert report.answer_quality.average_groundedness == 0.0
    assert report.answer_quality.average_completeness == 0.0
    assert report.answer_quality.average_overall == 0.0


def test_mismatched_answer_quality_score_count_raises_value_error():
    """Verify ValueError is raised when answer_quality_scores count differs from run results."""
    run_result = _make_run([_make_result(["c1"], ["c1"]), _make_result(["c2"], ["c2"])])
    # Only 1 score provided for 2 run results
    scores = [
        AnswerQualityScore(correctness=1.0, groundedness=1.0, completeness=1.0, overall=1.0)
    ]

    builder = EvaluationReportBuilder()
    with pytest.raises(ValueError, match="Mismatch between number of answer quality scores"):
        builder.build(run_result, scores, retrieval_k=1)


def test_invalid_retrieval_k_raises_error():
    """Verify invalid retrieval_k is rejected by RetrievalEvaluator through builder."""
    run_result = _make_run([_make_result(["c1"], ["c1"])])
    scores = [
        AnswerQualityScore(correctness=1.0, groundedness=1.0, completeness=1.0, overall=1.0)
    ]

    builder = EvaluationReportBuilder()
    with pytest.raises(ValueError, match="k must be a positive integer"):
        builder.build(run_result, scores, retrieval_k=0)


def test_retrieval_metrics_obtained_from_retrieval_evaluator(monkeypatch):
    """Verify builder delegates retrieval evaluation to RetrievalEvaluator."""
    called = []

    class MockRetrievalEvaluator:
        def evaluate(self, run_result, k):
            called.append((run_result, k))
            return RetrievalEvaluation(
                k=k,
                hit_rate=0.77,
                recall=0.88,
                mrr=0.99,
                total_cases=len(run_result.results),
            )

    run_result = _make_run([_make_result(["c1"], ["c1"])])
    scores = [
        AnswerQualityScore(correctness=1.0, groundedness=1.0, completeness=1.0, overall=1.0)
    ]

    builder = EvaluationReportBuilder(retrieval_evaluator=MockRetrievalEvaluator())
    report = builder.build(run_result, scores, retrieval_k=4)

    assert len(called) == 1
    assert called[0][1] == 4
    assert report.retrieval.hit_rate == pytest.approx(0.77)
    assert report.retrieval.recall == pytest.approx(0.88)
    assert report.retrieval.mrr == pytest.approx(0.99)


@pytest.mark.parametrize(
    "invalid_aggregate",
    [
        {"average_correctness": -0.1, "average_groundedness": 0.5, "average_completeness": 0.5, "average_overall": 0.5, "total_cases": 1},
        {"average_correctness": 1.1, "average_groundedness": 0.5, "average_completeness": 0.5, "average_overall": 0.5, "total_cases": 1},
        {"average_correctness": 0.5, "average_groundedness": 0.5, "average_completeness": 0.5, "average_overall": 0.5, "total_cases": -1},
    ],
)
def test_answer_quality_aggregate_score_bounds(invalid_aggregate):
    """Verify AnswerQualityAggregate bounds are strictly enforced between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        AnswerQualityAggregate(**invalid_aggregate)


def test_build_from_results_convenience_method():
    """Verify build_from_results evaluates answer quality and constructs report."""
    run_result = _make_run(
        [
            _make_result(
                retrieved_chunk_ids=["chunk-a"],
                relevant_chunk_ids=["chunk-a"],
                question="What is the refund period?",
                expected_answer="30 days.",
                actual_answer="30 days.",
            )
        ]
    )

    builder = EvaluationReportBuilder()
    report = builder.build_from_results(
        run_result=run_result,
        retrieval_k=1,
        contexts_list=[["Refunds are accepted within 30 days."]],
    )

    assert isinstance(report, EvaluationReport)
    assert report.total_cases == 1
    assert report.retrieval.hit_rate == pytest.approx(1.0)
    assert report.answer_quality.average_correctness == pytest.approx(1.0)
    assert report.answer_quality.average_groundedness == pytest.approx(1.0)


def test_integration_full_report_composition():
    """Integration test combining an EvaluationRunResult, retrieval metrics, and answer quality into one report."""
    results = [
        _make_result(
            retrieved_chunk_ids=["doc-1", "doc-2"],
            relevant_chunk_ids=["doc-1"],
            question="What is topic A?",
            expected_answer="Topic A details.",
            actual_answer="Topic A details provided.",
        ),
        _make_result(
            retrieved_chunk_ids=["wrong-1", "doc-3"],
            relevant_chunk_ids=["doc-3"],
            question="What is topic B?",
            expected_answer="Topic B information.",
            actual_answer="Topic B information.",
        ),
    ]
    run_result = _make_run(results)

    scores = [
        AnswerQualityScore(
            correctness=0.9,
            groundedness=0.8,
            completeness=0.85,
            overall=0.85,
            explanation="Case 1 evaluated.",
        ),
        AnswerQualityScore(
            correctness=1.0,
            groundedness=0.9,
            completeness=1.0,
            overall=0.9667,
            explanation="Case 2 evaluated.",
        ),
    ]

    builder = EvaluationReportBuilder()
    report = builder.build(
        run_result=run_result,
        answer_quality_scores=scores,
        retrieval_k=2,
    )

    # Retrieval assertions
    # Case 1: doc-1 at rank 1 -> hit=1, recall=1, rr=1.0
    # Case 2: doc-3 at rank 2 -> hit=1, recall=1, rr=0.5
    # Hit rate = 1.0, Recall = 1.0, MRR = (1.0 + 0.5) / 2 = 0.75
    assert report.retrieval.k == 2
    assert report.retrieval.hit_rate == pytest.approx(1.0)
    assert report.retrieval.recall == pytest.approx(1.0)
    assert report.retrieval.mrr == pytest.approx(0.75)
    assert report.retrieval.total_cases == 2

    # Answer quality assertions
    assert report.answer_quality.average_correctness == pytest.approx((0.9 + 1.0) / 2)
    assert report.answer_quality.average_groundedness == pytest.approx((0.8 + 0.9) / 2)
    assert report.answer_quality.average_completeness == pytest.approx((0.85 + 1.0) / 2)
    assert report.answer_quality.average_overall == pytest.approx((0.85 + 0.9667) / 2)
    assert report.answer_quality.total_cases == 2

    # Overall report
    assert report.total_cases == 2
