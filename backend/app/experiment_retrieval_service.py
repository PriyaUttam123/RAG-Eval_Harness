"""Retrieval strategy selection for experiments."""

from typing import Any, List, Optional

try:
    from backend.app.reranker import RerankerCandidate
except ImportError:
    from app.reranker import RerankerCandidate


class ExperimentRetrievalService:
    """Select and execute the configured retrieval strategy."""

    def __init__(
        self,
        vector_retriever: Any,
        hybrid_retriever: Any,
        reranking_service: Optional[Any] = None,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranking_service = reranking_service

    def retrieve(
        self,
        query: str,
        retrieval_strategy: str,
        retrieval_top_k: int,
        reranking_enabled: bool = False,
        reranking_top_k: Optional[int] = None,
    ) -> List[Any]:
        """Retrieve results using the configured strategy."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if retrieval_strategy not in {"vector", "hybrid"}:
            raise ValueError(
                "retrieval_strategy must be 'vector' or 'hybrid'."
            )

        if isinstance(retrieval_top_k, bool) or retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than zero.")

        cleaned_query = query.strip()

        if retrieval_strategy == "vector":
            results = self.vector_retriever.retrieve(
                query=cleaned_query,
                top_k=retrieval_top_k,
            )
        else:
            results = self.hybrid_retriever.retrieve(
                query=cleaned_query,
                top_k=retrieval_top_k,
            )

        if not reranking_enabled:
            return results

        if self.reranking_service is None:
            raise ValueError(
                "A reranking service is required when reranking is enabled."
            )

        candidates = self._to_candidates(results)

        if not candidates:
            return []

        final_top_k = (
            reranking_top_k
            if reranking_top_k is not None
            else retrieval_top_k
        )

        if isinstance(final_top_k, bool) or final_top_k <= 0:
            raise ValueError("reranking_top_k must be greater than zero.")

        return self.reranking_service.rerank(
            query=cleaned_query,
            candidates=candidates,
            top_k=final_top_k,
        )

    @staticmethod
    def _to_candidates(results: Any) -> List[RerankerCandidate]:
        """Convert retrieval results into reranker candidates."""

        if hasattr(results, "results"):
            items = results.results
        elif isinstance(results, list):
            items = results
        else:
            raise ValueError("Unsupported retrieval result format.")

        candidates = []

        for item in items:
            if hasattr(item, "distance"):
                retrieval_score = -float(item.distance)
            elif hasattr(item, "score"):
                retrieval_score = float(item.score)
            else:
                raise ValueError(
                    "Retrieved item must provide distance or score."
                )

            candidates.append(
                RerankerCandidate(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    metadata=item.metadata,
                    retrieval_score=retrieval_score,
                )
            )

        return candidates