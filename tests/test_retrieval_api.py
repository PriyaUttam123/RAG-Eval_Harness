from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.routes import retrieval as retrieval_route


client = TestClient(app)


class FakeRetrievedChunk:
    def __init__(self, chunk_id, text, metadata, distance):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.distance = distance


class FakeRetrievalResult:
    def model_dump(self):
        return {
            "query": "What is the refund policy?",
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "text": "The refund period is 30 days.",
                    "metadata": {
                        "source_filename": "policy.txt",
                        "chunk_index": 0,
                    },
                    "distance": 0.1,
                }
            ],
        }


class FakeRetrievalService:
    def __init__(self):
        self.received_query = None
        self.received_top_k = None

    def retrieve(self, query, top_k=None):
        self.received_query = query
        self.received_top_k = top_k
        return FakeRetrievalResult()


def test_search_returns_retrieved_chunks(monkeypatch):
    fake_service = FakeRetrievalService()

    monkeypatch.setattr(
        retrieval_route,
        "RetrievalService",
        lambda: fake_service,
    )

    response = client.post(
        "/retrieval/search",
        json={
            "query": "What is the refund policy?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is the refund policy?"
    assert len(data["results"]) == 1
    assert data["results"][0]["chunk_id"] == "chunk-1"
    assert data["results"][0]["text"] == "The refund period is 30 days."
    assert data["results"][0]["distance"] == 0.1

    assert fake_service.received_query == "What is the refund policy?"
    assert fake_service.received_top_k == 3


def test_search_allows_top_k_to_be_omitted(monkeypatch):
    fake_service = FakeRetrievalService()

    monkeypatch.setattr(
        retrieval_route,
        "RetrievalService",
        lambda: fake_service,
    )

    response = client.post(
        "/retrieval/search",
        json={
            "query": "What is the refund policy?",
        },
    )

    assert response.status_code == 200
    assert fake_service.received_top_k is None


def test_search_rejects_empty_query():
    response = client.post(
        "/retrieval/search",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422


def test_search_rejects_invalid_top_k():
    response = client.post(
        "/retrieval/search",
        json={
            "query": "What is the refund policy?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_search_returns_bad_request_for_value_error(monkeypatch):
    class FailingRetrievalService:
        def retrieve(self, query, top_k=None):
            raise ValueError("Query must be a non-empty string.")

    monkeypatch.setattr(
        retrieval_route,
        "RetrievalService",
        FailingRetrievalService,
    )

    response = client.post(
        "/retrieval/search",
        json={
            "query": "test",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Query must be a non-empty string."


def test_search_returns_server_error_for_runtime_error(monkeypatch):
    class FailingRetrievalService:
        def retrieve(self, query, top_k=None):
            raise RuntimeError("Embedding model is unavailable.")

    monkeypatch.setattr(
        retrieval_route,
        "RetrievalService",
        FailingRetrievalService,
    )

    response = client.post(
        "/retrieval/search",
        json={
            "query": "test",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Embedding model is unavailable."