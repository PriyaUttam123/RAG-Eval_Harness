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
from backend.app.experiment_models import ExperimentResult
from backend.app.retrieval_evaluator import RetrievalEvaluation


def make_config():
    return ExperimentConfig(
        name="baseline",
        chunk_size=512,
        chunk_overlap=50,
        retrieval_top_k=5,
        evaluation_k=3,
        embedding_model="all-MiniLM-L6-v2",
    )


def make_metadata():
    return ExperimentRunMetadata(
        experiment_id="exp-001",
        config=make_config(),
        dataset_name="golden",
        created_at=datetime.now(timezone.utc),
    )


def make_report():
    retrieval = RetrievalEvaluation(
        k=3,
        hit_rate=0.0,
        recall=0.0,
        mrr=0.0,
        total_cases=0,
    )

    answer_quality = AnswerQualityAggregate(
        average_correctness=0.0,
        average_groundedness=0.0,
        average_completeness=0.0,
        average_overall=0.0,
        total_cases=0,
    )

    return EvaluationReport(
        retrieval=retrieval,
        answer_quality=answer_quality,
        total_cases=0,
    )


def test_experiment_result_accepts_valid_data():
    result = ExperimentResult(
        run_metadata=make_metadata(),
        report=make_report(),
    )

    assert result.run_metadata.experiment_id == "exp-001"
    assert result.run_metadata.dataset_name == "golden"
    assert result.report.total_cases == 0
    assert result.error_count == 0
    assert result.duration_seconds is None


def test_experiment_result_accepts_duration():
    result = ExperimentResult(
        run_metadata=make_metadata(),
        report=make_report(),
        duration_seconds=1.25,
    )

    assert result.duration_seconds == 1.25


def test_experiment_result_rejects_negative_duration():
    with pytest.raises(ValueError):
        ExperimentResult(
            run_metadata=make_metadata(),
            report=make_report(),
            duration_seconds=-1,
        )


def test_experiment_result_rejects_negative_error_count():
    with pytest.raises(ValueError):
        ExperimentResult(
            run_metadata=make_metadata(),
            report=make_report(),
            error_count=-1,
        )


def test_experiment_result_preserves_custom_metadata():
    result = ExperimentResult(
        run_metadata=make_metadata(),
        report=make_report(),
        metadata={
            "retrieval_strategy": "hybrid",
            "reranking_enabled": True,
        },
    )

    assert result.metadata["retrieval_strategy"] == "hybrid"
    assert result.metadata["reranking_enabled"] is True
