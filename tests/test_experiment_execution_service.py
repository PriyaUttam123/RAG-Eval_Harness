
from datetime import datetime, timezone

import pytest

from backend.app.evaluation_report import (
    AnswerQualityAggregate,
    EvaluationReport,
)
from backend.app.experiment_config import (
    ExperimentConfig,
    ExperimentRunMetadata,
)
from backend.app.experiment_execution_service import (
    ExperimentExecutionService,
)
from backend.app.experiment_models import ExperimentResult
from backend.app.retrieval_evaluator import RetrievalEvaluation


def make_config():
    return ExperimentConfig(
        name="baseline",
        chunk_size=512,
        chunk_overlap=50,
        retrieval_top_k=5,
        evaluation_k=3,
        retrieval_strategy="vector",
        reranking_enabled=False,
        embedding_model="all-MiniLM-L6-v2",
        generation_model="test-model",
    )


def make_report():
    retrieval = RetrievalEvaluation(
        k=3,
        hit_rate=0.75,
        recall=0.5,
        mrr=0.6,
        total_cases=4,
    )

    answer_quality = AnswerQualityAggregate(
        average_correctness=0.8,
        average_groundedness=0.7,
        average_completeness=0.9,
        average_overall=0.8,
        total_cases=4,
    )

    return EvaluationReport(
        retrieval=retrieval,
        answer_quality=answer_quality,
        total_cases=4,
    )


class FakeEvaluationService:
    def __init__(self, report=None, error=None):
        self.report = report or make_report()
        self.error = error
        self.calls = []

    def run_dataset(self, dataset_name, retrieval_k, top_k):
        self.calls.append(
            {
                "dataset_name": dataset_name,
                "retrieval_k": retrieval_k,
                "top_k": top_k,
            }
        )

        if self.error is not None:
            raise self.error

        return self.report


class FakeExperimentService:
    def __init__(self):
        self.calls = []

    def create_run_metadata(
        self,
        experiment_id,
        config,
        dataset_name,
    ):
        self.calls.append(
            {
                "experiment_id": experiment_id,
                "config": config,
                "dataset_name": dataset_name,
            }
        )

        return ExperimentRunMetadata(
            experiment_id=experiment_id,
            config=config,
            dataset_name=dataset_name,
            created_at=datetime.now(timezone.utc),
        )


def test_run_returns_experiment_result():
    evaluation_service = FakeEvaluationService()
    experiment_service = FakeExperimentService()

    service = ExperimentExecutionService(
        evaluation_service=evaluation_service,
        experiment_service=experiment_service,
    )

    result = service.run(
        experiment_id="exp-001",
        config=make_config(),
    )

    assert isinstance(result, ExperimentResult)
    assert result.run_metadata.experiment_id == "exp-001"
    assert result.run_metadata.dataset_name == "golden"
    assert result.report.total_cases == 4
    assert result.error_count == 0


def test_run_injects_dataset_name_and_evaluation_parameters():
    evaluation_service = FakeEvaluationService()
    experiment_service = FakeExperimentService()

    service = ExperimentExecutionService(
        evaluation_service=evaluation_service,
        experiment_service=experiment_service,
    )

    config = make_config()

    service.run(
        experiment_id="exp-002",
        config=config,
        dataset_name="holdout",
    )

    assert evaluation_service.calls == [
        {
            "dataset_name": "holdout",
            "retrieval_k": 3,
            "top_k": 5,
        }
    ]


def test_run_creates_experiment_metadata():
    evaluation_service = FakeEvaluationService()
    experiment_service = FakeExperimentService()

    service = ExperimentExecutionService(
        evaluation_service=evaluation_service,
        experiment_service=experiment_service,
    )

    config = make_config()

    result = service.run(
        experiment_id="exp-003",
        config=config,
        dataset_name="holdout",
    )

    assert len(experiment_service.calls) == 1
    assert experiment_service.calls[0]["experiment_id"] == "exp-003"
    assert experiment_service.calls[0]["config"] == config
    assert experiment_service.calls[0]["dataset_name"] == "holdout"
    assert result.run_metadata.config == config


def test_run_records_duration():
    evaluation_service = FakeEvaluationService()
    experiment_service = FakeExperimentService()

    service = ExperimentExecutionService(
        evaluation_service=evaluation_service,
        experiment_service=experiment_service,
    )

    result = service.run(
        experiment_id="exp-004",
        config=make_config(),
    )

    assert result.duration_seconds is not None
    assert result.duration_seconds >= 0.0


def test_run_preserves_configuration_metadata():
    evaluation_service = FakeEvaluationService()
    experiment_service = FakeExperimentService()

    service = ExperimentExecutionService(
        evaluation_service=evaluation_service,
        experiment_service=experiment_service,
    )

    config = make_config()
    config.retrieval_strategy = "hybrid"
    config.reranking_enabled = True

    result = service.run(
        experiment_id="exp-005",
        config=config,
    )

    assert result.metadata["retrieval_strategy"] == "hybrid"
    assert result.metadata["reranking_enabled"] is True
    assert result.metadata["embedding_model"] == "all-MiniLM-L6-v2"
    assert result.metadata["generation_model"] == "test-model"


def test_run_records_execution_failure():
    evaluation_service = FakeEvaluationService(
        error=RuntimeError("evaluation failed")
    )
    experiment_service = FakeExperimentService()

    service = ExperimentExecutionService(
        evaluation_service=evaluation_service,
        experiment_service=experiment_service,
    )

    result = service.run(
        experiment_id="exp-006",
        config=make_config(),
    )

    assert isinstance(result, ExperimentResult)
    assert result.error_count == 1
    assert result.report.total_cases == 0
    assert result.duration_seconds is not None
    assert result.duration_seconds >= 0.0


def test_run_keeps_run_metadata_when_execution_fails():
    evaluation_service = FakeEvaluationService(
        error=RuntimeError("evaluation failed")
    )
    experiment_service = FakeExperimentService()

    service = ExperimentExecutionService(
        evaluation_service=evaluation_service,
        experiment_service=experiment_service,
    )

    result = service.run(
        experiment_id="exp-007",
        config=make_config(),
        dataset_name="holdout",
    )

    assert result.run_metadata.experiment_id == "exp-007"
    assert result.run_metadata.dataset_name == "holdout"
