"""Unit tests for EvaluationService end-to-end evaluation orchestration."""

import pytest

from backend.app.answer_quality_evaluator import (
    AnswerQualityEvaluator,
    AnswerQualityScore,
)
from backend.app.evaluation_dataset_manager import EvaluationDatasetManager
from backend.app.evaluation_models import EvaluationCase, EvaluationDataset
from backend.app.evaluation_report import (
    AnswerQualityAggregate,
    EvaluationReport,
    EvaluationReportBuilder,
)
from backend.app.evaluation_runner import (
    EvaluationResult,
    EvaluationRunner,
    EvaluationRunResult,
)
from backend.app.evaluation_service import EvaluationService
from backend.app.retrieval_evaluator import RetrievalEvaluation, RetrievalEvaluator


class FakeDatasetManager:
    """Fake dataset manager for testing without disk I/O."""

    def __init__(self, golden_dataset=None, holdout_dataset=None):
        self.golden_dataset = golden_dataset or EvaluationDataset(cases=[])
        self.holdout_dataset = holdout_dataset or EvaluationDataset(cases=[])
        self.loaded = []

    def load_golden(self):
        self.loaded.append("golden")
        return self.golden_dataset

    def load_holdout(self):
        self.loaded.append("holdout")
        return self.holdout_dataset


class FakeRunner:
    """Fake runner for recording run execution parameters."""

    def __init__(self, run_result=None):
        self.run_result = run_result
        self.last_dataset = None
        self.last_top_k = None

    def run(self, dataset, top_k=None):
        self.last_dataset = dataset
        self.last_top_k = top_k
        if self.run_result is not None:
            return self.run_result
        return EvaluationRunResult(
            results=[],
            total_cases=0,
            successful_cases=0,
            failed_cases=0,
        )


class FakeAnswerQualityEvaluator:
    """Fake answer quality evaluator for tracking case-by-case inputs."""

    def __init__(self, default_score=None):
        self.default_score = default_score or AnswerQualityScore(
            correctness=0.9,
            groundedness=0.8,
            completeness=0.85,
            overall=0.85,
            explanation="Fake score",
        )
        self.calls = []

    def evaluate_case(self, question, expected_answer, actual_answer, context):
        self.calls.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "context": context,
            }
        )
        return self.default_score


def _make_sample_result(
    question="What is RAG?",
    expected="Retrieval-Augmented Generation.",
    actual="RAG stands for Retrieval-Augmented Generation.",
    retrieved_chunk_ids=None,
    retrieved_contexts=None,
    error=None,
) -> EvaluationResult:
    return EvaluationResult(
        question=question,
        expected_answer=expected,
        actual_answer=actual,
        retrieved_chunk_ids=retrieved_chunk_ids if retrieved_chunk_ids is not None else ["chunk-1"],
        relevant_chunk_ids=["chunk-1"],
        case_type="factual",
        error=error,
        retrieved_contexts=retrieved_contexts if retrieved_contexts is not None else ["Context text about RAG."],
    )


def test_golden_dataset_evaluation():
    """Verify loading and evaluating the golden dataset."""
    golden_dataset = EvaluationDataset(
        cases=[
            EvaluationCase(
                question="What is RAG?",
                expected_answer="Retrieval-Augmented Generation.",
                relevant_chunk_ids=["chunk-1"],
                case_type="factual",
            )
        ]
    )
    dataset_mgr = FakeDatasetManager(golden_dataset=golden_dataset)
    runner = FakeRunner(
        run_result=EvaluationRunResult(
            results=[_make_sample_result()],
            total_cases=1,
            successful_cases=1,
            failed_cases=0,
        )
    )
    aq_evaluator = FakeAnswerQualityEvaluator()

    service = EvaluationService(
        dataset_manager=dataset_mgr,
        runner=runner,
        answer_quality_evaluator=aq_evaluator,
    )

    report = service.run_dataset("golden", retrieval_k=1, top_k=2)

    assert isinstance(report, EvaluationReport)
    assert dataset_mgr.loaded == ["golden"]
    assert runner.last_top_k == 2
    assert report.total_cases == 1
    assert report.retrieval.k == 1


def test_holdout_dataset_evaluation():
    """Verify loading and evaluating the holdout dataset."""
    holdout_dataset = EvaluationDataset(
        cases=[
            EvaluationCase(
                question="Holdout Q?",
                expected_answer="Holdout A.",
                relevant_chunk_ids=["chunk-2"],
                case_type="factual",
            )
        ]
    )
    dataset_mgr = FakeDatasetManager(holdout_dataset=holdout_dataset)
    runner = FakeRunner(
        run_result=EvaluationRunResult(
            results=[
                _make_sample_result(
                    question="Holdout Q?",
                    expected="Holdout A.",
                    actual="Holdout A.",
                    retrieved_chunk_ids=["chunk-2"],
                )
            ],
            total_cases=1,
            successful_cases=1,
            failed_cases=0,
        )
    )
    aq_evaluator = FakeAnswerQualityEvaluator()

    service = EvaluationService(
        dataset_manager=dataset_mgr,
        runner=runner,
        answer_quality_evaluator=aq_evaluator,
    )

    report = service.run_dataset("holdout", retrieval_k=2)

    assert isinstance(report, EvaluationReport)
    assert dataset_mgr.loaded == ["holdout"]
    assert report.total_cases == 1
    assert report.retrieval.k == 2


@pytest.mark.parametrize("invalid_name", ["unknown", "test", "", "   ", None])
def test_invalid_dataset_name_rejected(invalid_name):
    """Verify invalid or unsupported dataset names raise ValueError."""
    service = EvaluationService(
        dataset_manager=FakeDatasetManager(),
        runner=FakeRunner(),
    )

    with pytest.raises(ValueError):
        service.run_dataset(invalid_name, retrieval_k=1)


def test_top_k_forwarded_to_evaluation_runner():
    """Verify top_k parameter is forwarded to runner.run()."""
    runner = FakeRunner()
    service = EvaluationService(
        dataset_manager=FakeDatasetManager(),
        runner=runner,
    )

    service.run_dataset("golden", retrieval_k=3, top_k=7)
    assert runner.last_top_k == 7

    service.run_dataset("golden", retrieval_k=3, top_k=None)
    assert runner.last_top_k is None


def test_retrieval_k_forwarded_to_report_generation():
    """Verify retrieval_k is used as the cutoff for retrieval evaluation."""
    runner = FakeRunner(
        run_result=EvaluationRunResult(
            results=[_make_sample_result()],
            total_cases=1,
            successful_cases=1,
            failed_cases=0,
        )
    )
    service = EvaluationService(
        dataset_manager=FakeDatasetManager(),
        runner=runner,
    )

    report = service.run_dataset("golden", retrieval_k=5)
    assert report.retrieval.k == 5


def test_answer_quality_evaluator_called_for_each_case():
    """Verify AnswerQualityEvaluator is called once for each evaluation case."""
    results = [
        _make_result(q="Q1", exp="E1", act="A1", ctx=["Context 1"]),
        _make_result(q="Q2", exp="E2", act="A2", ctx=["Context 2"]),
        _make_result(q="Q3", exp="E3", act="A3", ctx=["Context 3"]),
    ]
    runner = FakeRunner(
        run_result=EvaluationRunResult(
            results=results,
            total_cases=3,
            successful_cases=3,
            failed_cases=0,
        )
    )
    aq_evaluator = FakeAnswerQualityEvaluator()

    service = EvaluationService(
        dataset_manager=FakeDatasetManager(),
        runner=runner,
        answer_quality_evaluator=aq_evaluator,
    )

    service.run_dataset("golden", retrieval_k=1)

    assert len(aq_evaluator.calls) == 3
    assert [c["question"] for c in aq_evaluator.calls] == ["Q1", "Q2", "Q3"]


def test_correct_context_passed_to_answer_quality_evaluator():
    """Verify retrieved context texts from EvaluationResult are forwarded to AQ evaluator."""
    result = _make_sample_result(
        question="Query",
        expected="Expected",
        actual="Actual",
        retrieved_contexts=["Snippet A", "Snippet B"],
    )
    runner = FakeRunner(
        run_result=EvaluationRunResult(
            results=[result],
            total_cases=1,
            successful_cases=1,
            failed_cases=0,
        )
    )
    aq_evaluator = FakeAnswerQualityEvaluator()

    service = EvaluationService(
        dataset_manager=FakeDatasetManager(),
        runner=runner,
        answer_quality_evaluator=aq_evaluator,
    )

    service.run_dataset("golden", retrieval_k=2)

    assert len(aq_evaluator.calls) == 1
    assert aq_evaluator.calls[0]["context"] == ["Snippet A", "Snippet B"]


def test_empty_dataset_produces_zero_report():
    """Verify evaluating an empty dataset produces a valid zero-valued report."""
    service = EvaluationService(
        dataset_manager=FakeDatasetManager(),
        runner=FakeRunner(),
    )

    report = service.run_dataset("golden", retrieval_k=3)

    assert report.total_cases == 0
    assert report.retrieval.total_cases == 0
    assert report.retrieval.hit_rate == 0.0
    assert report.answer_quality.total_cases == 0
    assert report.answer_quality.average_overall == 0.0


def test_individual_failed_case_does_not_crash_service():
    """Verify a failed RAG case with an error records empty outputs without crashing the service."""
    results = [
        _make_sample_result(question="Q1", expected="E1", actual="A1"),
        _make_sample_result(
            question="Q2",
            expected="E2",
            actual="",
            retrieved_chunk_ids=[],
            retrieved_contexts=[],
            error="RAG timeout error",
        ),
    ]
    runner = FakeRunner(
        run_result=EvaluationRunResult(
            results=results,
            total_cases=2,
            successful_cases=1,
            failed_cases=1,
        )
    )
    aq_evaluator = FakeAnswerQualityEvaluator()

    service = EvaluationService(
        dataset_manager=FakeDatasetManager(),
        runner=runner,
        answer_quality_evaluator=aq_evaluator,
    )

    report = service.run_dataset("golden", retrieval_k=1)

    assert report.total_cases == 2
    assert len(aq_evaluator.calls) == 2
    assert aq_evaluator.calls[1]["actual_answer"] == ""
    assert aq_evaluator.calls[1]["context"] == []


def test_dependency_injection_works():
    """Verify all subcomponents can be cleanly injected and defaults initialized safely."""
    service = EvaluationService()

    assert isinstance(service.dataset_manager, EvaluationDatasetManager)
    assert isinstance(service.runner, EvaluationRunner)
    assert isinstance(service.retrieval_evaluator, RetrievalEvaluator)
    assert isinstance(service.answer_quality_evaluator, AnswerQualityEvaluator)
    assert isinstance(service.report_builder, EvaluationReportBuilder)


def _make_result(q, exp, act, ctx) -> EvaluationResult:
    return EvaluationResult(
        question=q,
        expected_answer=exp,
        actual_answer=act,
        retrieved_chunk_ids=["chunk"],
        relevant_chunk_ids=["chunk"],
        case_type="factual",
        error=None,
        retrieved_contexts=ctx,
    )
