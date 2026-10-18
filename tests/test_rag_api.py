"""Unit tests for the RAG answer API route."""

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import rag as rag_route
from backend.app.main import app

client = TestClient(app)


class FakeRAGAnswer:
    def __init__(
        self,
        question: str = "What is the refund policy?",
        answer: str = "The refund period is 30 days.",
        sources: list = None,
    ):
        self.question = question
        self.answer = answer
        self.sources = sources or [
            {
                "chunk_id": "chunk-101",
                "text": "The refund period is 30 days.",
                "source_filename": "policy.txt",
                "chunk_index": 0,
                "distance": 0.12,
            }
        ]

    def model_dump(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
        }


class FakeRAGService:
    def __init__(self, answer_result=None, error=None):
        self.answer_result = answer_result or FakeRAGAnswer()
        self.error = error
        self.received_question = None
        self.received_top_k = None

    def answer(self, question: str, top_k=None):
        self.received_question = question
        self.received_top_k = top_k
        if self.error:
            raise self.error
        return self.answer_result


def test_answer_success_returns_question_answer_and_sources(monkeypatch):
    """Verify POST /rag/answer returns question, answer, and sources list."""
    fake_service = FakeRAGService()
    monkeypatch.setattr(rag_route, "RAGService", lambda: fake_service)

    response = client.post(
        "/rag/answer",
        json={
            "question": "What is the refund policy?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["question"] == "What is the refund policy?"
    assert data["answer"] == "The refund period is 30 days."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == "chunk-101"
    assert data["sources"][0]["text"] == "The refund period is 30 days."
    assert data["sources"][0]["source_filename"] == "policy.txt"
    assert data["sources"][0]["chunk_index"] == 0
    assert data["sources"][0]["distance"] == 0.12

    assert fake_service.received_question == "What is the refund policy?"
    assert fake_service.received_top_k == 3


def test_answer_allows_top_k_to_be_omitted(monkeypatch):
    """Verify top_k is optional and defaults to None."""
    fake_service = FakeRAGService()
    monkeypatch.setattr(rag_route, "RAGService", lambda: fake_service)

    response = client.post(
        "/rag/answer",
        json={
            "question": "What is the refund policy?",
        },
    )

    assert response.status_code == 200
    assert fake_service.received_top_k is None


def test_answer_forwards_top_k_correctly(monkeypatch):
    """Verify top_k parameter is forwarded to RAGService."""
    fake_service = FakeRAGService()
    monkeypatch.setattr(rag_route, "RAGService", lambda: fake_service)

    response = client.post(
        "/rag/answer",
        json={
            "question": "How to deploy?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert fake_service.received_top_k == 5


def test_answer_preserves_multiple_source_records(monkeypatch):
    """Verify returned source records preserve full metadata and ordering."""
    sources = [
        {
            "chunk_id": "c-1",
            "text": "First context",
            "source_filename": "manual.pdf",
            "chunk_index": 2,
            "distance": 0.05,
        },
        {
            "chunk_id": "c-2",
            "text": "Second context",
            "source_filename": "notes.txt",
            "chunk_index": 5,
            "distance": 0.22,
        },
    ]
    fake_answer = FakeRAGAnswer(sources=sources)
    fake_service = FakeRAGService(answer_result=fake_answer)
    monkeypatch.setattr(rag_route, "RAGService", lambda: fake_service)

    response = client.post(
        "/rag/answer",
        json={"question": "Summarize features"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 2
    assert data["sources"][0]["chunk_id"] == "c-1"
    assert data["sources"][0]["source_filename"] == "manual.pdf"
    assert data["sources"][1]["chunk_id"] == "c-2"
    assert data["sources"][1]["source_filename"] == "notes.txt"


def test_answer_rejects_empty_question():
    """Verify empty string question returns 422 Unprocessable Entity."""
    response = client.post(
        "/rag/answer",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_answer_handles_whitespace_only_question():
    """Verify whitespace-only question is rejected with 400 Bad Request via RAGService."""
    response = client.post(
        "/rag/answer",
        json={"question": "   \n\t  "},
    )

    assert response.status_code == 400
    assert "non-empty string" in response.json()["detail"]


@pytest.mark.parametrize("invalid_top_k", [0, -1, -5])
def test_answer_rejects_invalid_top_k(invalid_top_k):
    """Verify non-positive top_k returns 422 Unprocessable Entity."""
    response = client.post(
        "/rag/answer",
        json={
            "question": "What is the return policy?",
            "top_k": invalid_top_k,
        },
    )

    assert response.status_code == 422


def test_answer_returns_bad_request_for_value_error(monkeypatch):
    """Verify ValueError from RAGService becomes HTTP 400."""
    class FailingService:
        def answer(self, question, top_k=None):
            raise ValueError("Invalid question format provided.")

    monkeypatch.setattr(rag_route, "RAGService", FailingService)

    response = client.post(
        "/rag/answer",
        json={"question": "Valid input"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid question format provided."


def test_answer_returns_server_error_for_runtime_error(monkeypatch):
    """Verify RuntimeError from RAGService becomes HTTP 500."""
    class FailingService:
        def answer(self, question, top_k=None):
            raise RuntimeError("NVIDIA model failed to respond.")

    monkeypatch.setattr(rag_route, "RAGService", FailingService)

    response = client.post(
        "/rag/answer",
        json={"question": "Valid input"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "NVIDIA model failed to respond."


def test_openapi_schema_includes_rag_answer_path():
    """Verify the FastAPI OpenAPI schema includes /rag/answer route."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/rag/answer" in schema["paths"]
    assert "post" in schema["paths"]["/rag/answer"]
