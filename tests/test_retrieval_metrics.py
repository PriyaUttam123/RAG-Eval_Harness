import pytest

from backend.app.retrieval_metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_hit_rate_returns_one_when_relevant_result_is_in_top_k():
    assert hit_rate_at_k(
        ["chunk-3", "chunk-1", "chunk-5"],
        {"chunk-1"},
        2,
    ) == 1.0


def test_hit_rate_returns_zero_when_no_relevant_result_is_in_top_k():
    assert hit_rate_at_k(
        ["chunk-3", "chunk-5", "chunk-7"],
        {"chunk-1"},
        2,
    ) == 0.0


def test_recall_at_k_returns_fraction_of_relevant_results_found():
    assert recall_at_k(
        ["chunk-1", "chunk-4", "chunk-2"],
        {"chunk-1", "chunk-2", "chunk-3", "chunk-5"},
        3,
    ) == 0.5


def test_recall_at_k_returns_one_when_all_relevant_results_are_found():
    assert recall_at_k(
        ["chunk-1", "chunk-4", "chunk-2"],
        {"chunk-1", "chunk-2"},
        3,
    ) == 1.0


def test_reciprocal_rank_returns_reciprocal_of_first_relevant_rank():
    assert reciprocal_rank_at_k(
        ["chunk-7", "chunk-4", "chunk-2"],
        {"chunk-2"},
        3,
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_one_for_first_result():
    assert reciprocal_rank_at_k(
        ["chunk-2", "chunk-4", "chunk-7"],
        {"chunk-2"},
        3,
    ) == 1.0


def test_reciprocal_rank_returns_zero_when_no_relevant_result_exists():
    assert reciprocal_rank_at_k(
        ["chunk-7", "chunk-4", "chunk-9"],
        {"chunk-2"},
        3,
    ) == 0.0


def test_mean_reciprocal_rank_averages_query_scores():
    assert mean_reciprocal_rank(
        [
            ["chunk-1", "chunk-2"],
            ["chunk-3", "chunk-4"],
            ["chunk-5", "chunk-6"],
        ],
        [
            {"chunk-1"},
            {"chunk-4"},
            {"chunk-9"},
        ],
        2,
    ) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_mean_reciprocal_rank_returns_zero_for_empty_input():
    assert mean_reciprocal_rank([], [], 3) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_non_positive_k(k):
    with pytest.raises(ValueError):
        hit_rate_at_k(["chunk-1"], {"chunk-1"}, k)

    with pytest.raises(ValueError):
        recall_at_k(["chunk-1"], {"chunk-1"}, k)

    with pytest.raises(ValueError):
        reciprocal_rank_at_k(["chunk-1"], {"chunk-1"}, k)


def test_mean_reciprocal_rank_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError):
        mean_reciprocal_rank(
            [["chunk-1"]],
            [{"chunk-1"}, {"chunk-2"}],
            3,
        )


def test_metrics_return_zero_when_no_relevant_ids_exist():
    assert hit_rate_at_k(["chunk-1"], set(), 3) == 0.0
    assert recall_at_k(["chunk-1"], set(), 3) == 0.0
    assert reciprocal_rank_at_k(["chunk-1"], set(), 3) == 0.0