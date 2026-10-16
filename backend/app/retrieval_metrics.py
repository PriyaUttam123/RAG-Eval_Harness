"""Metrics for evaluating retrieval quality."""

from typing import Iterable, List, Sequence, Set


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")


def _validate_ranked_ids(retrieved_ids: Sequence[str]) -> None:
    if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in retrieved_ids):
        raise ValueError("Retrieved chunk IDs must be non-empty strings.")


def _validate_relevant_ids(relevant_ids: Iterable[str]) -> Set[str]:
    relevant_set = set(relevant_ids)

    if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in relevant_set):
        raise ValueError("Relevant chunk IDs must be non-empty strings.")

    return relevant_set


def hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Return 1.0 if any relevant chunk appears in the top-k results."""
    _validate_k(k)
    _validate_ranked_ids(retrieved_ids)
    relevant_set = _validate_relevant_ids(relevant_ids)

    if not relevant_set:
        return 0.0

    top_k_ids = retrieved_ids[:k]

    return 1.0 if any(chunk_id in relevant_set for chunk_id in top_k_ids) else 0.0


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Return the fraction of relevant chunks found in the top-k results."""
    _validate_k(k)
    _validate_ranked_ids(retrieved_ids)
    relevant_set = _validate_relevant_ids(relevant_ids)

    if not relevant_set:
        return 0.0

    top_k_ids = set(retrieved_ids[:k])
    retrieved_relevant = top_k_ids.intersection(relevant_set)

    return len(retrieved_relevant) / len(relevant_set)


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Return the reciprocal rank of the first relevant result in top-k."""
    _validate_k(k)
    _validate_ranked_ids(retrieved_ids)
    relevant_set = _validate_relevant_ids(relevant_ids)

    if not relevant_set:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    ranked_results: Sequence[Sequence[str]],
    relevant_results: Sequence[Iterable[str]],
    k: int,
) -> float:
    """Return the mean reciprocal rank across multiple evaluation queries."""
    _validate_k(k)

    if len(ranked_results) != len(relevant_results):
        raise ValueError(
            "ranked_results and relevant_results must have the same length."
        )

    if not ranked_results:
        return 0.0

    reciprocal_ranks: List[float] = [
        reciprocal_rank_at_k(retrieved_ids, relevant_ids, k)
        for retrieved_ids, relevant_ids in zip(ranked_results, relevant_results)
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)