import pytest

from backend.app.reranker import RerankedResult, RerankerCandidate
from backend.app.reranking_service import RerankingService


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append(
            {
                "query": query,
                "candidates": candidates,
                "top_k": top_k,
            }
        )

        return [
            RerankedResult(
                chunk_id="reranked-1",
                text="Reranked text",
                metadata={"source": "test.txt"},
                score=0.95,
                original_rank=0,
            )
        ]


def test_reranking_service_accepts_injected_reranker():
    reranker = FakeReranker()

    service = RerankingService(reranker=reranker)

    assert service.reranker is reranker


def test_reranking_service_delegates_to_reranker():
    reranker = FakeReranker()
    service = RerankingService(reranker=reranker)

    candidates = [
        RerankerCandidate(
            chunk_id="chunk-1",
            text="Python programming",
            retrieval_score=0.8,
        )
    ]

    results = service.rerank(
        query="Python",
        candidates=candidates,
        top_k=1,
    )

    assert len(reranker.calls) == 1
    assert reranker.calls[0]["query"] == "Python"
    assert reranker.calls[0]["candidates"] == candidates
    assert reranker.calls[0]["top_k"] == 1

    assert len(results) == 1
    assert results[0].chunk_id == "reranked-1"


def test_reranking_service_preserves_reranker_result():
    reranker = FakeReranker()
    service = RerankingService(reranker=reranker)

    results = service.rerank(
        query="Python",
        candidates=[],
        top_k=5,
    )

    assert results[0].metadata == {"source": "test.txt"}
    assert results[0].score == 0.95


def test_reranking_service_propagates_validation_errors():
    class ValidatingFakeReranker:
        def rerank(self, query, candidates, top_k):
            raise ValueError("invalid reranking request")

    service = RerankingService(
        reranker=ValidatingFakeReranker()
    )

    with pytest.raises(ValueError, match="invalid reranking request"):
        service.rerank(
            query="Python",
            candidates=[],
            top_k=1,
        )
