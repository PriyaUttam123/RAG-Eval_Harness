import pytest

from backend.app.reranker import RerankedResult
from backend.app.reranked_retrieval import RerankedRetrievalService


class FakeRetrievedChunk:
    def __init__(self, chunk_id, text, metadata, distance):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.distance = distance


class FakeRetrievalResult:
    def __init__(self, results):
        self.results = results


class FakeRetrievalService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
            }
        )
        return self.result


class FakeRerankingService:
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
                chunk_id="chunk-2",
                text="Better result",
                metadata={"source": "test.txt"},
                score=0.9,
                original_rank=1,
            )
        ]


def test_retrieval_happens_before_reranking():
    retrieval = FakeRetrievalService(
        FakeRetrievalResult(
            [
                FakeRetrievedChunk(
                    "chunk-1",
                    "Python text",
                    {"source": "a.txt"},
                    0.2,
                )
            ]
        )
    )
    reranking = FakeRerankingService()

    service = RerankedRetrievalService(retrieval, reranking)

    results = service.retrieve(
        query="Python",
        retrieval_top_k=5,
        final_top_k=2,
    )

    assert retrieval.calls == [
        {
            "query": "Python",
            "top_k": 5,
        }
    ]

    assert len(reranking.calls) == 1
    assert reranking.calls[0]["query"] == "Python"
    assert reranking.calls[0]["top_k"] == 2
    assert len(reranking.calls[0]["candidates"]) == 1

    assert results[0].chunk_id == "chunk-2"


def test_vector_distance_is_converted_to_retrieval_score():
    retrieval = FakeRetrievalService(
        FakeRetrievalResult(
            [
                FakeRetrievedChunk(
                    "chunk-1",
                    "Python text",
                    {},
                    0.25,
                )
            ]
        )
    )
    reranking = FakeRerankingService()

    service = RerankedRetrievalService(retrieval, reranking)

    service.retrieve(
        query="Python",
        retrieval_top_k=5,
        final_top_k=2,
    )

    candidate = reranking.calls[0]["candidates"][0]

    assert candidate.retrieval_score == -0.25


def test_empty_retrieval_does_not_call_reranker():
    retrieval = FakeRetrievalService(
        FakeRetrievalResult([])
    )
    reranking = FakeRerankingService()

    service = RerankedRetrievalService(retrieval, reranking)

    results = service.retrieve(
        query="Python",
        retrieval_top_k=5,
        final_top_k=2,
    )

    assert results == []
    assert reranking.calls == []


def test_query_is_trimmed():
    retrieval = FakeRetrievalService(
        FakeRetrievalResult([])
    )
    reranking = FakeRerankingService()

    service = RerankedRetrievalService(retrieval, reranking)

    service.retrieve(
        query="  Python  ",
        retrieval_top_k=5,
        final_top_k=2,
    )

    assert retrieval.calls[0]["query"] == "Python"


@pytest.mark.parametrize(
    ("query", "retrieval_top_k", "final_top_k"),
    [
        ("", 5, 2),
        ("   ", 5, 2),
        ("Python", 0, 2),
        ("Python", -1, 2),
        ("Python", 5, 0),
        ("Python", 5, -1),
    ],
)
def test_invalid_retrieval_arguments_are_rejected(
    query,
    retrieval_top_k,
    final_top_k,
):
    retrieval = FakeRetrievalService(
        FakeRetrievalResult([])
    )
    reranking = FakeRerankingService()

    service = RerankedRetrievalService(retrieval, reranking)

    with pytest.raises(ValueError):
        service.retrieve(
            query=query,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
        )


def test_reranker_errors_propagate():
    retrieval = FakeRetrievalService(
        FakeRetrievalResult(
            [
                FakeRetrievedChunk(
                    "chunk-1",
                    "Python text",
                    {},
                    0.2,
                )
            ]
        )
    )

    class FailingRerankingService:
        def rerank(self, query, candidates, top_k):
            raise RuntimeError("reranking failed")

    service = RerankedRetrievalService(
        retrieval,
        FailingRerankingService(),
    )

    with pytest.raises(RuntimeError, match="reranking failed"):
        service.retrieve(
            query="Python",
            retrieval_top_k=5,
            final_top_k=2,
        )