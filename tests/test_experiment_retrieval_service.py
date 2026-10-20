from types import SimpleNamespace

import pytest

from backend.app.experiment_retrieval_service import ExperimentRetrievalService


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
            }
        )
        return self.results


class FakeRerankingService:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append(
            {
                "query": query,
                "candidates": candidates,
                "top_k": top_k,
            }
        )
        return self.results


def make_results():
    return [
        SimpleNamespace(
            chunk_id="chunk-1",
            text="First chunk",
            metadata={"source": "doc.txt"},
            distance=0.2,
        ),
        SimpleNamespace(
            chunk_id="chunk-2",
            text="Second chunk",
            metadata={"source": "doc.txt"},
            distance=0.4,
        ),
    ]


def test_vector_strategy_uses_vector_retriever():
    vector = FakeRetriever(make_results())
    hybrid = FakeRetriever([])
    service = ExperimentRetrievalService(vector, hybrid)

    result = service.retrieve(
        query="  test query  ",
        retrieval_strategy="vector",
        retrieval_top_k=5,
    )

    assert result == vector.results
    assert vector.calls == [{"query": "test query", "top_k": 5}]
    assert hybrid.calls == []


def test_hybrid_strategy_uses_hybrid_retriever():
    vector = FakeRetriever([])
    hybrid = FakeRetriever(make_results())
    service = ExperimentRetrievalService(vector, hybrid)

    result = service.retrieve(
        query="test query",
        retrieval_strategy="hybrid",
        retrieval_top_k=6,
    )

    assert result == hybrid.results
    assert hybrid.calls == [{"query": "test query", "top_k": 6}]
    assert vector.calls == []


def test_reranking_is_skipped_when_disabled():
    vector = FakeRetriever(make_results())
    hybrid = FakeRetriever([])
    reranker = FakeRerankingService([])
    service = ExperimentRetrievalService(vector, hybrid, reranker)

    result = service.retrieve(
        query="test query",
        retrieval_strategy="vector",
        retrieval_top_k=4,
        reranking_enabled=False,
    )

    assert result == vector.results
    assert reranker.calls == []


def test_reranking_receives_retrieved_candidates():
    vector = FakeRetriever(make_results())
    hybrid = FakeRetriever([])
    reranked_results = ["reranked"]
    reranker = FakeRerankingService(reranked_results)

    service = ExperimentRetrievalService(vector, hybrid, reranker)

    result = service.retrieve(
        query="test query",
        retrieval_strategy="vector",
        retrieval_top_k=4,
        reranking_enabled=True,
        reranking_top_k=2,
    )

    assert result == reranked_results
    assert len(reranker.calls) == 1

    call = reranker.calls[0]
    assert call["query"] == "test query"
    assert call["top_k"] == 2
    assert len(call["candidates"]) == 2
    assert call["candidates"][0].chunk_id == "chunk-1"
    assert call["candidates"][0].retrieval_score == pytest.approx(-0.2)


def test_reranking_defaults_to_retrieval_top_k():
    vector = FakeRetriever(make_results())
    hybrid = FakeRetriever([])
    reranker = FakeRerankingService(["reranked"])

    service = ExperimentRetrievalService(vector, hybrid, reranker)

    service.retrieve(
        query="test query",
        retrieval_strategy="vector",
        retrieval_top_k=7,
        reranking_enabled=True,
    )

    assert reranker.calls[0]["top_k"] == 7


def test_invalid_strategy_is_rejected():
    service = ExperimentRetrievalService(
        FakeRetriever([]),
        FakeRetriever([]),
    )

    with pytest.raises(ValueError, match="retrieval_strategy"):
        service.retrieve(
            query="test query",
            retrieval_strategy="unknown",
            retrieval_top_k=4,
        )


def test_reranking_requires_reranking_service():
    service = ExperimentRetrievalService(
        FakeRetriever(make_results()),
        FakeRetriever([]),
    )

    with pytest.raises(ValueError, match="reranking service"):
        service.retrieve(
            query="test query",
            retrieval_strategy="vector",
            retrieval_top_k=4,
            reranking_enabled=True,
        )


def test_empty_results_do_not_call_reranker():
    vector = FakeRetriever([])
    hybrid = FakeRetriever([])
    reranker = FakeRerankingService(["should not be returned"])

    service = ExperimentRetrievalService(vector, hybrid, reranker)

    result = service.retrieve(
        query="test query",
        retrieval_strategy="vector",
        retrieval_top_k=4,
        reranking_enabled=True,
    )

    assert result == []
    assert reranker.calls == []