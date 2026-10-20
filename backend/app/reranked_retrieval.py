from typing import Any, List

try:
    from backend.app.reranker import RerankedResult, RerankerCandidate
    from backend.app.reranking_service import RerankingService
except ImportError:
    from app.reranker import RerankedResult, RerankerCandidate
    from app.reranking_service import RerankingService


class RerankedRetrievalService:
    """Retrieves candidate chunks and reranks them."""

    def __init__(
        self,
        retrieval_service: Any,
        reranking_service: RerankingService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.reranking_service = reranking_service

    def retrieve(
        self,
        query: str,
        retrieval_top_k: int,
        final_top_k: int,
    ) -> List[RerankedResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if isinstance(retrieval_top_k, bool) or retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than zero.")

        if isinstance(final_top_k, bool) or final_top_k <= 0:
            raise ValueError("final_top_k must be greater than zero.")

        retrieval_result = self.retrieval_service.retrieve(
            query=query.strip(),
            top_k=retrieval_top_k,
        )

        candidates = self._to_candidates(retrieval_result)

        if not candidates:
            return []

        return self.reranking_service.rerank(
            query=query.strip(),
            candidates=candidates,
            top_k=final_top_k,
        )

    @staticmethod
    def _to_candidates(retrieval_result: Any) -> List[RerankerCandidate]:
        if hasattr(retrieval_result, "results"):
            retrieved_items = retrieval_result.results
        elif isinstance(retrieval_result, list):
            retrieved_items = retrieval_result
        else:
            raise ValueError("Unsupported retrieval result format.")

        candidates = []

        for item in retrieved_items:
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