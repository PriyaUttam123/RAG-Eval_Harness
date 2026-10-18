"""Evaluator service for computing retrieval quality metrics on benchmark results."""

from typing import List

from pydantic import BaseModel, Field

try:
    from backend.app.evaluation_runner import EvaluationResult, EvaluationRunResult
    from backend.app.retrieval_metrics import (
        _validate_k,
        hit_rate_at_k,
        mean_reciprocal_rank,
        recall_at_k,
    )
except ImportError:
    from app.evaluation_runner import EvaluationResult, EvaluationRunResult
    from app.retrieval_metrics import (
        _validate_k,
        hit_rate_at_k,
        mean_reciprocal_rank,
        recall_at_k,
    )


class RetrievalEvaluation(BaseModel):
    """Aggregated retrieval benchmark metrics for a specific top-k cutoff."""

    k: int = Field(..., gt=0, description="Top-k evaluation cutoff parameter.")
    hit_rate: float = Field(..., ge=0.0, le=1.0, description="Mean Hit Rate@k across all cases.")
    recall: float = Field(..., ge=0.0, le=1.0, description="Mean Recall@k across all cases.")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank@k across all cases.")
    total_cases: int = Field(..., ge=0, description="Total number of evaluated cases.")


class RetrievalEvaluator:
    """Computes retrieval quality metrics from evaluation run results."""

    def evaluate(
        self,
        run_result: EvaluationRunResult,
        k: int,
    ) -> RetrievalEvaluation:
        """Calculate Hit Rate@k, Recall@k, and MRR@k across evaluation run results.

        Args:
            run_result: Results container from an evaluation run.
            k: Cutoff depth for retrieval metrics evaluation.

        Returns:
            RetrievalEvaluation containing mean hit rate, recall, MRR, and case count.

        Raises:
            ValueError: If k is not a positive integer or run_result is invalid.
        """
        _validate_k(k)

        if run_result is None or not isinstance(run_result, EvaluationRunResult):
            raise ValueError("run_result must be a valid EvaluationRunResult instance.")

        results: List[EvaluationResult] = run_result.results
        total_cases = len(results)

        if total_cases == 0:
            return RetrievalEvaluation(
                k=k,
                hit_rate=0.0,
                recall=0.0,
                mrr=0.0,
                total_cases=0,
            )

        hit_rates: List[float] = [
            hit_rate_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k)
            for r in results
        ]

        recalls: List[float] = [
            recall_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k)
            for r in results
        ]

        ranked_results = [r.retrieved_chunk_ids for r in results]
        relevant_results = [r.relevant_chunk_ids for r in results]
        mrr = mean_reciprocal_rank(ranked_results, relevant_results, k)

        mean_hit_rate = sum(hit_rates) / total_cases
        mean_recall = sum(recalls) / total_cases

        return RetrievalEvaluation(
            k=k,
            hit_rate=float(mean_hit_rate),
            recall=float(mean_recall),
            mrr=float(mrr),
            total_cases=total_cases,
        )
