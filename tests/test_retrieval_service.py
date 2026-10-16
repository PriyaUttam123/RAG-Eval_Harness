import pytest

from backend.app.retrieval_service import RetrievalService


class FakeEmbeddingService:
    def __init__(self):
        self.received_query = None

    def embed_text(self, text):
        self.received_query = text
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self):
        self.received_embedding = None
        self.received_top_k = None

    def search(self, query_embedding, top_k=None):
        self.received_embedding = query_embedding
        self.received_top_k = top_k

        return {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [
                [
                    "The refund period is 30 days.",
                    "Refunds are processed within five business days.",
                ]
            ],
            "metadatas": [
                [
                    {
                        "source_filename": "policy.txt",
                        "chunk_index": 0,
                    },
                    {
                        "source_filename": "policy.txt",
                        "chunk_index": 1,
                    },
                ]
            ],
            "distances": [[0.1, 0.25]],
        }


def test_retrieve_returns_clean_results():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = service.retrieve("  What is the refund period?  ", top_k=2)

    assert result.query == "What is the refund period?"
    assert len(result.results) == 2

    assert result.results[0].chunk_id == "chunk-1"
    assert result.results[0].text == "The refund period is 30 days."
    assert result.results[0].metadata["source_filename"] == "policy.txt"
    assert result.results[0].distance == 0.1

    assert result.results[1].chunk_id == "chunk-2"
    assert result.results[1].distance == 0.25


def test_retrieve_strips_query_before_embedding():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    service.retrieve("   hello world   ")

    assert embedding_service.received_query == "hello world"


def test_retrieve_passes_embedding_and_top_k_to_vector_store():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    service.retrieve("test query", top_k=5)

    assert vector_store.received_embedding == [0.1, 0.2, 0.3]
    assert vector_store.received_top_k == 5


def test_retrieve_rejects_empty_query():
    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        service.retrieve("")


def test_retrieve_rejects_whitespace_query():
    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        service.retrieve("   ")


def test_retrieve_handles_empty_search_results():
    class EmptyVectorStore:
        def search(self, query_embedding, top_k=None):
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=EmptyVectorStore(),
    )

    result = service.retrieve("unknown question")

    assert result.query == "unknown question"
    assert result.results == []