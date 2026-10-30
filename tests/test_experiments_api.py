"""Unit tests for the experiment execution API routes."""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import evaluation as evaluation_route
from backend.app.evaluation_report import (
    AnswerQualityAggregate,
    EvaluationReport,
)
from backend.app.experiment_config import (
    ExperimentConfig,
    ExperimentRunMetadata,
)
from backend.app.experiment_models import ExperimentResult
from backend.app.failure_analysis import EvaluationFailure, FailureAnalysis
from backend.app.main import app
from backend.app.retrieval_evaluator import RetrievalEvaluation

client = TestClient(app)


def _make_valid_config_payload(
    name: str = "Vector baseline",
    retrieval_strategy: str = "vector",
    reranking_enabled: bool = False,
) -> dict:
    return {
        "name": name,
        "description": "Baseline configuration for testing",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "retrieval_top_k": 5,
        "evaluation_k": 3,
        "retrieval_strategy": retrieval_strategy,
        "reranking_enabled": reranking_enabled,
        "embedding_model": "all-MiniLM-L6-v2",
        "generation_model": "test-generation-model",
    }


def _make_sample_experiment_result(
    experiment_id: str,
    config: ExperimentConfig,
    dataset_name: str = "golden",
    include_failure: bool = False,
) -> ExperimentResult:
    failures = []
    if include_failure:
        failures.append(
            EvaluationFailure(
                case_id="case-001",
                question="What is the policy?",
                expected_answer="Refund within 30 days.",
                actual_answer="No refund.",
                failure_type="answer_quality",
                answer_quality_score=0.25,
            )
        )

    report = EvaluationReport(
        retrieval=RetrievalEvaluation(
            k=config.evaluation_k,
            hit_rate=0.8,
            recall=0.75,
            mrr=0.85,
            total_cases=2,
        ),
        answer_quality=AnswerQualityAggregate(
            average_correctness=0.9,
            average_groundedness=0.85,
            average_completeness=0.8,
            average_overall=0.85,
            total_cases=2,
        ),
        failure_analysis=FailureAnalysis(
            failures=failures,
            total_failures=len(failures),
        ),
        total_cases=2,
    )

    metadata = ExperimentRunMetadata(
        experiment_id=experiment_id,
        config=config,
        dataset_name=dataset_name,
        created_at=datetime.now(timezone.utc),
    )

    return ExperimentResult(
        run_metadata=metadata,
        report=report,
        duration_seconds=1.42,
        error_count=0,
        metadata={
            "retrieval_strategy": config.retrieval_strategy,
            "reranking_enabled": config.reranking_enabled,
            "embedding_model": config.embedding_model,
            "generation_model": config.generation_model,
        },
    )


class FakeExperimentExecutionService:
    def __init__(self, result=None, results=None, error=None):
        self.result = result
        self.results = results
        self.error = error
        self.run_calls = []
        self.run_many_calls = []

    def run(self, experiment_id, config, dataset_name="golden"):
        self.run_calls.append({
            "experiment_id": experiment_id,
            "config": config,
            "dataset_name": dataset_name,
        })
        if self.error:
            raise self.error
        if dataset_name not in {"golden", "holdout"}:
            raise ValueError(f"Invalid dataset_name '{dataset_name}'. Must be 'golden' or 'holdout'.")
        if self.result is not None:
            return self.result
        return _make_sample_experiment_result(experiment_id, config, dataset_name)

    def run_many(self, experiments, dataset_name="golden"):
        self.run_many_calls.append({
            "experiments": experiments,
            "dataset_name": dataset_name,
        })
        if self.error:
            raise self.error
        if dataset_name not in {"golden", "holdout"}:
            raise ValueError(f"Invalid dataset_name '{dataset_name}'. Must be 'golden' or 'holdout'.")
        if self.results is not None:
            return self.results
        return [
            _make_sample_experiment_result(exp_id, cfg, dataset_name)
            for exp_id, cfg in experiments
        ]


def test_single_experiment_success(monkeypatch):
    """Verify POST /evaluation/experiments/run succeeds with valid payload."""
    fake_service = FakeExperimentExecutionService()
    monkeypatch.setattr(evaluation_route, "ExperimentExecutionService", lambda: fake_service)

    payload = {
        "experiment_id": "exp-001",
        "config": _make_valid_config_payload(),
        "dataset_name": "golden",
    }

    response = client.post("/evaluation/experiments/run", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["run_metadata"]["experiment_id"] == "exp-001"
    assert data["run_metadata"]["dataset_name"] == "golden"
    assert data["run_metadata"]["config"]["name"] == "Vector baseline"
    assert data["report"]["total_cases"] == 2
    assert "retrieval" in data["report"]
    assert "answer_quality" in data["report"]
    assert "failure_analysis" in data["report"]
    assert data["duration_seconds"] == pytest.approx(1.42)
    assert data["error_count"] == 0
    assert data["metadata"]["retrieval_strategy"] == "vector"

    assert len(fake_service.run_calls) == 1
    assert fake_service.run_calls[0]["experiment_id"] == "exp-001"
    assert fake_service.run_calls[0]["dataset_name"] == "golden"
    assert isinstance(fake_service.run_calls[0]["config"], ExperimentConfig)


def test_single_experiment_default_dataset_is_golden(monkeypatch):
    """Verify dataset_name defaults to 'golden' when omitted."""
    fake_service = FakeExperimentExecutionService()
    monkeypatch.setattr(evaluation_route, "ExperimentExecutionService", lambda: fake_service)

    payload = {
        "experiment_id": "exp-default",
        "config": _make_valid_config_payload(),
    }

    response = client.post("/evaluation/experiments/run", json=payload)

    assert response.status_code == 200
    assert fake_service.run_calls[0]["dataset_name"] == "golden"


def test_single_experiment_dataset_alias(monkeypatch):
    """Verify 'dataset' is accepted as an alias for dataset_name."""
    fake_service = FakeExperimentExecutionService()
    monkeypatch.setattr(evaluation_route, "ExperimentExecutionService", lambda: fake_service)

    payload = {
        "experiment_id": "exp-holdout",
        "config": _make_valid_config_payload(),
        "dataset": "holdout",
    }

    response = client.post("/evaluation/experiments/run", json=payload)

    assert response.status_code == 200
    assert fake_service.run_calls[0]["dataset_name"] == "holdout"


def test_single_experiment_preserves_failure_analysis(monkeypatch):
    """Verify returned result retains detailed failure-analysis records."""
    sample_config = ExperimentConfig(**_make_valid_config_payload())
    result_with_failures = _make_sample_experiment_result(
        "exp-fail", sample_config, "golden", include_failure=True
    )
    fake_service = FakeExperimentExecutionService(result=result_with_failures)
    monkeypatch.setattr(evaluation_route, "ExperimentExecutionService", lambda: fake_service)

    response = client.post(
        "/evaluation/experiments/run",
        json={"experiment_id": "exp-fail", "config": _make_valid_config_payload()},
    )

    assert response.status_code == 200
    data = response.json()
    fa = data["report"]["failure_analysis"]
    assert fa["total_failures"] == 1
    assert len(fa["failures"]) == 1
    assert fa["failures"][0]["case_id"] == "case-001"
    assert fa["failures"][0]["failure_type"] == "answer_quality"


def test_batch_experiments_success_and_ordering(monkeypatch):
    """Verify POST /evaluation/experiments/run-batch preserves experiment order."""
    fake_service = FakeExperimentExecutionService()
    monkeypatch.setattr(evaluation_route, "ExperimentExecutionService", lambda: fake_service)

    payload = {
        "experiments": [
            {
                "experiment_id": "exp-vector",
                "config": _make_valid_config_payload(name="Vector", retrieval_strategy="vector"),
            },
            {
                "experiment_id": "exp-hybrid",
                "config": _make_valid_config_payload(name="Hybrid", retrieval_strategy="hybrid"),
            },
            {
                "experiment_id": "exp-reranked",
                "config": _make_valid_config_payload(name="Reranked", retrieval_strategy="hybrid", reranking_enabled=True),
            },
        ],
        "dataset_name": "golden",
    }

    response = client.post("/evaluation/experiments/run-batch", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 3
    assert [res["run_metadata"]["experiment_id"] for res in data] == [
        "exp-vector",
        "exp-hybrid",
        "exp-reranked",
    ]
    assert [res["run_metadata"]["config"]["name"] for res in data] == [
        "Vector",
        "Hybrid",
        "Reranked",
    ]

    assert len(fake_service.run_many_calls) == 1
    passed_experiments = fake_service.run_many_calls[0]["experiments"]
    assert [item[0] for item in passed_experiments] == [
        "exp-vector",
        "exp-hybrid",
        "exp-reranked",
    ]


def test_batch_experiments_empty_list(monkeypatch):
    """Verify batch endpoint accepts an empty list of experiments."""
    fake_service = FakeExperimentExecutionService()
    monkeypatch.setattr(evaluation_route, "ExperimentExecutionService", lambda: fake_service)

    response = client.post(
        "/evaluation/experiments/run-batch",
        json={"experiments": [], "dataset_name": "golden"},
    )

    assert response.status_code == 200
    assert response.json() == []


def test_single_experiment_unsupported_dataset_rejected():
    """Verify unsupported dataset name results in 400 Bad Request."""
    payload = {
        "experiment_id": "exp-001",
        "config": _make_valid_config_payload(),
        "dataset_name": "unsupported_dataset",
    }

    response = client.post("/evaluation/experiments/run", json=payload)

    assert response.status_code == 400
    assert "golden" in response.json()["detail"].lower()


def test_batch_experiments_unsupported_dataset_rejected():
    """Verify batch run with unsupported dataset results in 400 Bad Request."""
    payload = {
        "experiments": [
            {
                "experiment_id": "exp-001",
                "config": _make_valid_config_payload(),
            }
        ],
        "dataset_name": "invalid_dataset",
    }

    response = client.post("/evaluation/experiments/run-batch", json=payload)

    assert response.status_code == 400
    assert "golden" in response.json()["detail"].lower()


def test_single_experiment_invalid_config_validation_error():
    """Verify invalid config parameters return 422 Unprocessable Entity."""
    # chunk_overlap must be strictly less than chunk_size
    invalid_config = _make_valid_config_payload()
    invalid_config["chunk_size"] = 100
    invalid_config["chunk_overlap"] = 150

    response = client.post(
        "/evaluation/experiments/run",
        json={"experiment_id": "exp-001", "config": invalid_config},
    )

    assert response.status_code == 422


def test_single_experiment_missing_required_fields():
    """Verify missing required fields return 422 Unprocessable Entity."""
    response = client.post("/evaluation/experiments/run", json={})
    assert response.status_code == 422


def test_batch_experiments_missing_required_fields():
    """Verify batch endpoint missing required fields returns 422."""
    response = client.post("/evaluation/experiments/run-batch", json={})
    assert response.status_code == 422


def test_single_experiment_runtime_error_returns_500(monkeypatch):
    """Verify RuntimeError during execution returns 500 Internal Server Error."""
    fake_service = FakeExperimentExecutionService(
        error=RuntimeError("Retrieval pipeline crashed")
    )
    monkeypatch.setattr(evaluation_route, "ExperimentExecutionService", lambda: fake_service)

    response = client.post(
        "/evaluation/experiments/run",
        json={"experiment_id": "exp-fail", "config": _make_valid_config_payload()},
    )

    assert response.status_code == 500
    assert "Retrieval pipeline crashed" in response.json()["detail"]


def test_batch_experiments_runtime_error_returns_500(monkeypatch):
    """Verify RuntimeError during batch execution returns 500."""
    fake_service = FakeExperimentExecutionService(
        error=RuntimeError("Batch execution failure")
    )
    monkeypatch.setattr(evaluation_route, "ExperimentExecutionService", lambda: fake_service)

    response = client.post(
        "/evaluation/experiments/run-batch",
        json={
            "experiments": [
                {"experiment_id": "exp-1", "config": _make_valid_config_payload()}
            ]
        },
    )

    assert response.status_code == 500
    assert "Batch execution failure" in response.json()["detail"]


def test_openapi_schema_includes_experiment_endpoints():
    """Verify experiment endpoints are registered in OpenAPI paths."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]

    assert "/evaluation/experiments/run" in paths
    assert "post" in paths["/evaluation/experiments/run"]

    assert "/evaluation/experiments/run-batch" in paths
    assert "post" in paths["/evaluation/experiments/run-batch"]
