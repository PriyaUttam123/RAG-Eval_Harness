"""Unit tests for RetrievalEvaluator service."""

import pytest

from backend.app.evaluation_runner import EvaluationResult, EvaluationRunResult
from backend.app.retrieval_evaluator import RetrievalEvaluation, RetrievalEvaluator
from backend.app import retrieval_evaluator as evaluator_module


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


def test_perfect_retrieval():
    """Verify metrics when all relevant chunks are retrieved at top ranks."""
    evaluator = RetrievalEvaluator()
    run_result = _make_run(
        [
            _make_result(
                retrieved_chunk_ids=["chunk-1", "chunk-2"],
                relevant_chunk_ids=["chunk-1", "chunk-2"],
            ),
            _make_result(
                retrieved_chunk_ids=["chunk-3"],
                relevant_chunk_ids=["chunk-3"],
            ),
        ]
    )

    evaluation = evaluator.evaluate(run_result, k=2)

    assert isinstance(evaluation, RetrievalEvaluation)
    assert evaluation.k == 2
    assert evaluation.total_cases == 2
    assert evaluation.hit_rate == pytest.approx(1.0)
    assert evaluation.recall == pytest.approx(1.0)
    assert evaluation.mrr == pytest.approx(1.0)


def test_complete_retrieval_miss():
    """Verify metrics when no relevant chunks are retrieved."""
    evaluator = RetrievalEvaluator()
    run_result = _make_run(
        [
            _make_result(
                retrieved_chunk_ids=["unrelated-1", "unrelated-2"],
                relevant_chunk_ids=["chunk-1"],
            ),
            _make_result(
                retrieved_chunk_ids=["unrelated-3"],
                relevant_chunk_ids=["chunk-2"],
            ),
        ]
    )

    evaluation = evaluator.evaluate(run_result, k=2)

    assert evaluation.hit_rate == pytest.approx(0.0)
    assert evaluation.recall == pytest.approx(0.0)
    assert evaluation.mrr == pytest.approx(0.0)
    assert evaluation.total_cases == 2


def test_partial_recall():
    """Verify recall is fractional when only a subset of relevant chunks are retrieved."""
    evaluator = RetrievalEvaluator()
    # Case with 4 relevant chunks, only 2 retrieved
    run_result = _make_run(
        [
            _make_result(
                retrieved_chunk_ids=["chunk-1", "chunk-2", "irrelevant"],
                relevant_chunk_ids=["chunk-1", "chunk-2", "chunk-3", "chunk-4"],
            )
        ]
    )

    evaluation = evaluator.evaluate(run_result, k=3)

    assert evaluation.hit_rate == pytest.approx(1.0)
    assert evaluation.recall == pytest.approx(2 / 4)
    assert evaluation.mrr == pytest.approx(1.0)


def test_different_first_relevant_ranks_affecting_mrr():
    """Verify MRR properly weights ranks of the first relevant result."""
    evaluator = RetrievalEvaluator()
    # Case 1: first relevant at rank 1 -> reciprocal rank = 1.0
    # Case 2: first relevant at rank 2 -> reciprocal rank = 0.5
    # Case 3: first relevant at rank 3 -> reciprocal rank = 1/3
    run_result = _make_run(
        [
            _make_result(
                retrieved_chunk_ids=["c-1", "x", "y"],
                relevant_chunk_ids=["c-1"],
            ),
            _make_result(
                retrieved_chunk_ids=["x", "c-2", "y"],
                relevant_chunk_ids=["c-2"],
            ),
            _make_result(
                retrieved_chunk_ids=["x", "y", "c-3"],
                relevant_chunk_ids=["c-3"],
            ),
        ]
    )

    evaluation = evaluator.evaluate(run_result, k=3)

    expected_mrr = (1.0 + 0.5 + (1.0 / 3.0)) / 3.0
    assert evaluation.mrr == pytest.approx(expected_mrr)
    assert evaluation.hit_rate == pytest.approx(1.0)


def test_multiple_evaluation_cases_averages():
    """Verify overall metrics correctly average across varied evaluation cases."""
    evaluator = RetrievalEvaluator()
    run_result = _make_run(
        [
            # Hit=1.0, Recall=1.0, RR=1.0
            _make_result(
                retrieved_chunk_ids=["c-1"],
                relevant_chunk_ids=["c-1"],
            ),
            # Hit=0.0, Recall=0.0, RR=0.0
            _make_result(
                retrieved_chunk_ids=["miss"],
                relevant_chunk_ids=["c-2"],
            ),
        ]
    )

    evaluation = evaluator.evaluate(run_result, k=1)

    assert evaluation.total_cases == 2
    assert evaluation.hit_rate == pytest.approx(0.5)
    assert evaluation.recall == pytest.approx(0.5)
    assert evaluation.mrr == pytest.approx(0.5)


def test_k_limiting_retrieved_results():
    """Verify results beyond k are excluded from metric calculations."""
    evaluator = RetrievalEvaluator()
    # Relevant chunk is at rank 4
    run_result = _make_run(
        [
            _make_result(
                retrieved_chunk_ids=["x1", "x2", "x3", "c-target"],
                relevant_chunk_ids=["c-target"],
            )
        ]
    )

    # When k=3, the relevant chunk is missed
    eval_k3 = evaluator.evaluate(run_result, k=3)
    assert eval_k3.hit_rate == pytest.approx(0.0)
    assert eval_k3.recall == pytest.approx(0.0)
    assert eval_k3.mrr == pytest.approx(0.0)

    # When k=4, the relevant chunk is included at rank 4
    eval_k4 = evaluator.evaluate(run_result, k=4)
    assert eval_k4.hit_rate == pytest.approx(1.0)
    assert eval_k4.recall == pytest.approx(1.0)
    assert eval_k4.mrr == pytest.approx(1.0 / 4.0)


def test_failed_evaluation_cases_produce_zero_retrieval_metrics():
    """Verify failed cases are counted in total_cases and contribute 0.0 scores."""
    evaluator = RetrievalEvaluator()
    run_result = _make_run(
        [
            # Successful case: Hit=1, Recall=1, RR=1
            _make_result(
                retrieved_chunk_ids=["c-1"],
                relevant_chunk_ids=["c-1"],
            ),
            # Failed case: empty retrieved_chunk_ids and error message
            _make_result(
                retrieved_chunk_ids=[],
                relevant_chunk_ids=["c-2"],
                actual_answer="",
                error="RAG generation timeout exception",
            ),
        ]
    )

    evaluation = evaluator.evaluate(run_result, k=1)

    assert evaluation.total_cases == 2
    assert evaluation.hit_rate == pytest.approx(0.5)
    assert evaluation.recall == pytest.approx(0.5)
    assert evaluation.mrr == pytest.approx(0.5)


def test_empty_evaluation_run():
    """Verify empty run_result returns zero metrics and total_cases=0."""
    evaluator = RetrievalEvaluator()
    run_result = _make_run([])

    evaluation = evaluator.evaluate(run_result, k=3)

    assert evaluation.k == 3
    assert evaluation.total_cases == 0
    assert evaluation.hit_rate == pytest.approx(0.0)
    assert evaluation.recall == pytest.approx(0.0)
    assert evaluation.mrr == pytest.approx(0.0)


def test_total_cases_matches_number_of_results():
    """Verify total_cases strictly matches the count of EvaluationResults."""
    evaluator = RetrievalEvaluator()
    cases = [
        _make_result(["c1"], ["c1"]),
        _make_result(["c2"], ["c2"]),
        _make_result(["c3"], ["c3"]),
        _make_result([], ["c4"], error="Failed"),
    ]
    run_result = _make_run(cases)

    evaluation = evaluator.evaluate(run_result, k=2)

    assert evaluation.total_cases == 4


@pytest.mark.parametrize("invalid_k", [0, -1, -5, "3", 1.5, True, False, None])
def test_invalid_k_rejected(invalid_k):
    """Verify non-positive or non-integer k raises ValueError."""
    evaluator = RetrievalEvaluator()
    run_result = _make_run([_make_result(["c-1"], ["c-1"])])

    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluator.evaluate(run_result, k=invalid_k)


def test_returned_metric_values_within_bounds():
    """Verify hit_rate, recall, and mrr are bounded within [0.0, 1.0]."""
    evaluator = RetrievalEvaluator()
    run_result = _make_run(
        [
            _make_result(["a", "b"], ["b"]),
            _make_result(["c"], ["c"]),
            _make_result(["d"], ["e"]),
        ]
    )

    evaluation = evaluator.evaluate(run_result, k=2)

    for metric in [evaluation.hit_rate, evaluation.recall, evaluation.mrr]:
        assert 0.0 <= metric <= 1.0


def test_evaluator_uses_existing_retrieval_metric_functions(monkeypatch):
    """Verify evaluator invokes existing metric functions rather than duplicating formulas."""
    called_metrics = []

    orig_hit = evaluator_module.hit_rate_at_k
    orig_rec = evaluator_module.recall_at_k
    orig_mrr = evaluator_module.mean_reciprocal_rank

    def spy_hit(retrieved, relevant, k):
        called_metrics.append("hit_rate_at_k")
        return orig_hit(retrieved, relevant, k)

    def spy_rec(retrieved, relevant, k):
        called_metrics.append("recall_at_k")
        return orig_rec(retrieved, relevant, k)

    def spy_mrr(ranked, relevant, k):
        called_metrics.append("mean_reciprocal_rank")
        return orig_mrr(ranked, relevant, k)

    monkeypatch.setattr(evaluator_module, "hit_rate_at_k", spy_hit)
    monkeypatch.setattr(evaluator_module, "recall_at_k", spy_rec)
    monkeypatch.setattr(evaluator_module, "mean_reciprocal_rank", spy_mrr)

    evaluator = RetrievalEvaluator()
    run_result = _make_run([_make_result(["c-1"], ["c-1"])])

    evaluator.evaluate(run_result, k=1)

    assert "hit_rate_at_k" in called_metrics
    assert "recall_at_k" in called_metrics
    assert "mean_reciprocal_rank" in called_metrics


@pytest.mark.parametrize("invalid_run", [None, "invalid", [1, 2, 3], {"results": []}])
def test_invalid_run_result_rejected(invalid_run):
    """Verify non-EvaluationRunResult argument raises ValueError."""
    evaluator = RetrievalEvaluator()

    with pytest.raises(ValueError, match="EvaluationRunResult"):
        evaluator.evaluate(invalid_run, k=2)
