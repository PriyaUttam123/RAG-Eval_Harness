from datetime import datetime, timezone

from backend.app.evaluation_report import (
    AnswerQualityAggregate,
    EvaluationReport,
)
from backend.app.experiment_comparison import ExperimentComparisonService
from backend.app.experiment_config import ExperimentConfig, ExperimentRunMetadata
from backend.app.experiment_models import ExperimentResult
from backend.app.retrieval_evaluator import RetrievalEvaluation


def make_result(
    experiment_id: str = "exp-1",
    name: str = "Vector baseline",
) -> ExperimentResult:
    config = ExperimentConfig(
        name=name,
        chunk_size=500,
        chunk_overlap=50,
        retrieval_top_k=5,
        evaluation_k=5,
        retrieval_strategy="vector",
        reranking_enabled=False,
        embedding_model="test-embedding",
        generation_model="test-generation",
    )

    metadata = ExperimentRunMetadata(
        experiment_id=experiment_id,
        config=config,
        dataset_name="golden",
        created_at=datetime.now(timezone.utc),
    )

    report = EvaluationReport(
        retrieval=RetrievalEvaluation(
            k=5,
            hit_rate=0.8,
            recall=0.7,
            mrr=0.6,
            total_cases=10,
        ),
        answer_quality=AnswerQualityAggregate(
            average_correctness=0.9,
            average_groundedness=0.8,
            average_completeness=0.7,
            average_overall=0.8,
            total_cases=10,
        ),
        total_cases=10,
    )

    return ExperimentResult(
        run_metadata=metadata,
        report=report,
        duration_seconds=1.5,
        error_count=0,
    )


def test_compare_single_experiment():
    service = ExperimentComparisonService()

    comparison = service.compare([make_result()])

    assert comparison.total_experiments == 1
    assert len(comparison.rows) == 1

    row = comparison.rows[0]
    assert row.experiment_id == "exp-1"
    assert row.experiment_name == "Vector baseline"
    assert row.hit_rate == 0.8
    assert row.recall == 0.7
    assert row.mrr == 0.6
    assert row.average_correctness == 0.9
    assert row.average_groundedness == 0.8
    assert row.average_completeness == 0.7
    assert row.average_overall == 0.8
    assert row.duration_seconds == 1.5
    assert row.error_count == 0


def test_compare_multiple_experiments_preserves_order():
    service = ExperimentComparisonService()

    first = make_result("exp-1", "Vector baseline")
    second = make_result("exp-2", "Hybrid retrieval")

    comparison = service.compare([first, second])

    assert comparison.total_experiments == 2
    assert [row.experiment_id for row in comparison.rows] == [
        "exp-1",
        "exp-2",
    ]
    assert [row.experiment_name for row in comparison.rows] == [
        "Vector baseline",
        "Hybrid retrieval",
    ]


def test_compare_empty_results():
    service = ExperimentComparisonService()

    comparison = service.compare([])

    assert comparison.total_experiments == 0
    assert comparison.rows == []
