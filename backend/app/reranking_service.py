from typing import List

try:
    from backend.app.reranker import (
        RerankedResult,
        Reranker,
        RerankerCandidate,
    )
except ImportError:
    from app.reranker import (
        RerankedResult,
        Reranker,
        RerankerCandidate,
    )


class RerankingService:
    """Service layer for reranking retrieved candidates."""

    def __init__(self, reranker: Reranker) -> None:
        self.reranker = reranker

    def rerank(
        self,
        query: str,
        candidates: List[RerankerCandidate],
        top_k: int,
    ) -> List[RerankedResult]:
        """Delegate reranking to the configured reranker."""
        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
