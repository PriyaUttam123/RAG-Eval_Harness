"""Unit tests for the evaluation API routes."""

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import evaluation as evaluation_route
from backend.app.evaluation_report import (
    AnswerQualityAggregate,
    EvaluationReport,
)
from backend.app.main import app
from backend.app.retrieval_evaluator import RetrievalEvaluation

client = TestClient(app)


def _make_sample_report(k: int = 2, total: int = 3) -> EvaluationReport:
    return EvaluationReport(
        retrieval=RetrievalEvaluation(
            k=k,
            hit_rate=0.8,
            recall=0.75,
            mrr=0.85,
            total_cases=total,
        ),
        answer_quality=AnswerQualityAggregate(
            average_correctness=0.9,
            average_groundedness=0.85,
            average_completeness=0.8,
            average_overall=0.85,
            total_cases=total,
        ),
        total_cases=total,
    )


class FakeEvaluationService:
    def __init__(self, report=None, error=None):
        self.report = report or _make_sample_report()
        self.error = error
        self.last_dataset_name = None
        self.last_retrieval_k = None
        self.last_top_k = None

    def run_dataset(self, dataset_name, retrieval_k, top_k=None):
        self.last_dataset_name = dataset_name
        self.last_retrieval_k = retrieval_k
        self.last_top_k = top_k
        if self.error:
            raise self.error
        if dataset_name not in {"golden", "holdout"}:
            raise ValueError(f"Unsupported dataset '{dataset_name}'.")
        return self.report


def test_golden_evaluation_success(monkeypatch):
    """Verify POST /evaluation/run succeeds with golden dataset."""
    fake_service = FakeEvaluationService()
    monkeypatch.setattr(evaluation_route, "EvaluationService", lambda: fake_service)

    response = client.post(
        "/evaluation/run",
        json={
            "dataset": "golden",
            "retrieval_k": 3,
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total_cases"] == 3
    assert "retrieval" in data
    assert "answer_quality" in data
    assert data["retrieval"]["k"] == 2
    assert fake_service.last_dataset_name == "golden"
    assert fake_service.last_retrieval_k == 3
    assert fake_service.last_top_k == 5


def test_holdout_evaluation_success(monkeypatch):
    """Verify POST /evaluation/run succeeds with holdout dataset."""
    fake_service = FakeEvaluationService()
    monkeypatch.setattr(evaluation_route, "EvaluationService", lambda: fake_service)

    response = client.post(
        "/evaluation/run",
        json={
            "dataset": "holdout",
            "retrieval_k": 2,
        },
    )

    assert response.status_code == 200
    assert fake_service.last_dataset_name == "holdout"
    assert fake_service.last_retrieval_k == 2
    assert fake_service.last_top_k is None


def test_parameter_forwarding(monkeypatch):
    """Verify retrieval_k and top_k are forwarded cleanly to the service."""
    fake_service = FakeEvaluationService()
    monkeypatch.setattr(evaluation_route, "EvaluationService", lambda: fake_service)

    response = client.post(
        "/evaluation/run",
        json={
            "dataset": "golden",
            "retrieval_k": 4,
            "top_k": 8,
        },
    )

    assert response.status_code == 200
    assert fake_service.last_retrieval_k == 4
    assert fake_service.last_top_k == 8


def test_invalid_dataset_name_returns_400(monkeypatch):
    """Verify unsupported dataset name returns HTTP 400."""
    fake_service = FakeEvaluationService()
    monkeypatch.setattr(evaluation_route, "EvaluationService", lambda: fake_service)

    response = client.post(
        "/evaluation/run",
        json={
            "dataset": "unsupported_dataset",
            "retrieval_k": 2,
        },
    )

    assert response.status_code == 400
    assert "Unsupported dataset" in response.json()["detail"]


@pytest.mark.parametrize("invalid_k", [0, -1, -5])
def test_invalid_retrieval_k_returns_422(invalid_k):
    """Verify non-positive retrieval_k returns 422 Unprocessable Entity."""
    response = client.post(
        "/evaluation/run",
        json={
            "dataset": "golden",
            "retrieval_k": invalid_k,
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("invalid_top_k", [0, -1])
def test_invalid_top_k_returns_422(invalid_top_k):
    """Verify non-positive top_k returns 422 Unprocessable Entity."""
    response = client.post(
        "/evaluation/run",
        json={
            "dataset": "golden",
            "retrieval_k": 2,
            "top_k": invalid_top_k,
        },
    )

    assert response.status_code == 422


def test_service_value_error_returns_400(monkeypatch):
    """Verify ValueError from EvaluationService becomes HTTP 400."""
    class FailingService:
        def run_dataset(self, dataset_name, retrieval_k, top_k=None):
            raise ValueError("Invalid evaluation configuration parameter.")

    monkeypatch.setattr(evaluation_route, "EvaluationService", FailingService)

    response = client.post(
        "/evaluation/run",
        json={
            "dataset": "golden",
            "retrieval_k": 2,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid evaluation configuration parameter."


def test_service_runtime_error_returns_500(monkeypatch):
    """Verify RuntimeError from EvaluationService becomes HTTP 500."""
    class FailingService:
        def run_dataset(self, dataset_name, retrieval_k, top_k=None):
            raise RuntimeError("Evaluation execution engine encountered an internal failure.")

    monkeypatch.setattr(evaluation_route, "EvaluationService", FailingService)

    response = client.post(
        "/evaluation/run",
        json={
            "dataset": "golden",
            "retrieval_k": 2,
        },
    )

    assert response.status_code == 500
    assert "internal failure" in response.json()["detail"]


def test_response_structure_matches_evaluation_report(monkeypatch):
    """Verify returned JSON contains all expected fields matching EvaluationReport schema."""
    fake_service = FakeEvaluationService()
    monkeypatch.setattr(evaluation_route, "EvaluationService", lambda: fake_service)

    response = client.post(
        "/evaluation/run",
        json={"dataset": "golden", "retrieval_k": 2},
    )

    assert response.status_code == 200
    data = response.json()

    assert "retrieval" in data
    assert "hit_rate" in data["retrieval"]
    assert "recall" in data["retrieval"]
    assert "mrr" in data["retrieval"]

    assert "answer_quality" in data
    assert "average_correctness" in data["answer_quality"]
    assert "average_groundedness" in data["answer_quality"]
    assert "average_completeness" in data["answer_quality"]
    assert "average_overall" in data["answer_quality"]


def test_openapi_paths_include_evaluation_run():
    """Verify /evaluation/run is registered in FastAPI OpenAPI schema."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/evaluation/run" in schema["paths"]
    assert "post" in schema["paths"]["/evaluation/run"]

def _make_experiment_result(
    experiment_id: str = "exp-1",
    name: str = "Vector baseline",
) -> dict:
    return {
        "run_metadata": {
            "experiment_id": experiment_id,
            "config": {
                "name": name,
                "description": "",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "retrieval_top_k": 5,
                "evaluation_k": 5,
                "retrieval_strategy": "vector",
                "reranking_enabled": False,
                "embedding_model": "test-embedding",
                "generation_model": "test-generation",
            },
            "dataset_name": "golden",
            "created_at": "2026-01-01T00:00:00Z",
        },
        "report": _make_sample_report().model_dump(),
        "duration_seconds": 1.5,
        "error_count": 0,
        "metadata": {},
    }


def test_compare_evaluations_success():
    """Verify POST /evaluation/compare returns comparison data."""
    response = client.post(
        "/evaluation/compare",
        json={
            "results": [
                _make_experiment_result("exp-1", "Vector baseline"),
                _make_experiment_result("exp-2", "Hybrid retrieval"),
            ]
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total_experiments"] == 2
    assert len(data["rows"]) == 2
    assert data["rows"][0]["experiment_id"] == "exp-1"
    assert data["rows"][1]["experiment_id"] == "exp-2"


def test_compare_evaluations_empty_results():
    """Verify comparison accepts an empty result list."""
    response = client.post(
        "/evaluation/compare",
        json={"results": []},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total_experiments"] == 0
    assert data["rows"] == []


def test_openapi_paths_include_evaluation_compare():
    """Verify /evaluation/compare is registered in FastAPI OpenAPI."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    assert "/evaluation/compare" in schema["paths"]
    assert "post" in schema["paths"]["/evaluation/compare"]
