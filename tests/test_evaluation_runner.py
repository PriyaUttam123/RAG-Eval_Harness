"""Unit tests for EvaluationRunner."""

from typing import Any, Dict, List, Optional, Set

import pytest

from backend.app.evaluation_models import EvaluationCase, EvaluationDataset
from backend.app.evaluation_runner import (
    EvaluationResult,
    EvaluationRunner,
    EvaluationRunResult,
)
from backend.app.rag_service import RAGService


class FakeRAGSource:
    """Fake source object mimicking RAGSource."""

    def __init__(self, chunk_id: str):
        self.chunk_id = chunk_id


class FakeRAGAnswer:
    """Fake answer object mimicking RAGAnswer."""

    def __init__(self, answer: str, sources: Optional[List[Any]] = None):
        self.answer = answer
        self.sources = sources or []


class FakeRAGService:
    """Fake RAG service that tracks calls and optionally simulates errors."""

    def __init__(
        self,
        answers_by_question: Optional[Dict[str, FakeRAGAnswer]] = None,
        default_answer: str = "Default answer.",
        default_sources: Optional[List[Any]] = None,
        fail_questions: Optional[Set[str]] = None,
    ):
        self.answers_by_question = answers_by_question or {}
        self.default_answer = default_answer
        self.default_sources = default_sources or [FakeRAGSource("default-chunk")]
        self.fail_questions = fail_questions or set()
        self.calls: List[Dict[str, Any]] = []

    def answer(self, question: str, top_k: Optional[int] = None) -> FakeRAGAnswer:
        self.calls.append({"question": question, "top_k": top_k})
        if question in self.fail_questions:
            raise RuntimeError(f"RAG failure for question: {question}")
        if question in self.answers_by_question:
            return self.answers_by_question[question]
        return FakeRAGAnswer(
            answer=f"Generated answer for: {question}",
            sources=self.default_sources,
        )


def _build_sample_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        cases=[
            EvaluationCase(
                question="What is the refund period?",
                expected_answer="30 days from purchase.",
                relevant_chunk_ids=["chunk-refund-1", "chunk-refund-2"],
                case_type="factual",
            ),
            EvaluationCase(
                question="How do I change shipping address?",
                expected_answer="Go to account settings before dispatch.",
                relevant_chunk_ids=["chunk-shipping-1"],
                case_type="factual",
            ),
            EvaluationCase(
                question="What is the warranty on spacecraft?",
                expected_answer="Spacecraft are not sold here.",
                relevant_chunk_ids=[],
                case_type="unanswerable",
            ),
        ]
    )


def test_successful_single_case_run():
    """Verify executing an evaluation run with a single case."""
    case = EvaluationCase(
        question="What is the return window?",
        expected_answer="Within 14 days.",
        relevant_chunk_ids=["chunk-ret-1"],
        case_type="factual",
    )
    dataset = EvaluationDataset(cases=[case])

    fake_rag = FakeRAGService(
        answers_by_question={
            "What is the return window?": FakeRAGAnswer(
                answer="Returns must occur within 14 days.",
                sources=[FakeRAGSource("chunk-ret-1"), FakeRAGSource("chunk-ret-2")],
            )
        }
    )

    runner = EvaluationRunner(rag_service=fake_rag)
    run_result = runner.run(dataset, top_k=2)

    assert isinstance(run_result, EvaluationRunResult)
    assert run_result.total_cases == 1
    assert run_result.successful_cases == 1
    assert run_result.failed_cases == 0

    assert len(run_result.results) == 1
    result = run_result.results[0]
    assert result.question == "What is the return window?"
    assert result.expected_answer == "Within 14 days."
    assert result.actual_answer == "Returns must occur within 14 days."
    assert result.retrieved_chunk_ids == ["chunk-ret-1", "chunk-ret-2"]
    assert result.relevant_chunk_ids == ["chunk-ret-1"]
    assert result.case_type == "factual"
    assert result.error is None


def test_successful_multiple_case_run():
    """Verify executing multiple cases correctly aggregates counts."""
    dataset = _build_sample_dataset()
    fake_rag = FakeRAGService()

    runner = EvaluationRunner(rag_service=fake_rag)
    run_result = runner.run(dataset)

    assert run_result.total_cases == 3
    assert run_result.successful_cases == 3
    assert run_result.failed_cases == 0
    assert len(run_result.results) == 3


def test_results_preserve_input_order():
    """Verify results in EvaluationRunResult preserve the dataset ordering."""
    dataset = _build_sample_dataset()
    fake_rag = FakeRAGService()

    runner = EvaluationRunner(rag_service=fake_rag)
    run_result = runner.run(dataset)

    questions = [r.question for r in run_result.results]
    expected_questions = [c.question for c in dataset.cases]
    assert questions == expected_questions


def test_question_and_expected_answer_preserved():
    """Verify question and expected answer are accurately recorded."""
    case = EvaluationCase(
        question="Specific question?",
        expected_answer="Exact expected answer.",
        case_type="factual",
    )
    dataset = EvaluationDataset(cases=[case])
    runner = EvaluationRunner(rag_service=FakeRAGService())

    run_result = runner.run(dataset)
    assert run_result.results[0].question == "Specific question?"
    assert run_result.results[0].expected_answer == "Exact expected answer."


def test_retrieved_chunk_ids_extracted_correctly():
    """Verify chunk IDs from RAGAnswer sources are extracted in order."""
    case = EvaluationCase(
        question="Check chunks?",
        expected_answer="Answer.",
        case_type="factual",
    )
    dataset = EvaluationDataset(cases=[case])

    fake_rag = FakeRAGService(
        default_sources=[
            FakeRAGSource("chunk-alpha"),
            FakeRAGSource("chunk-beta"),
            FakeRAGSource("chunk-gamma"),
        ]
    )
    runner = EvaluationRunner(rag_service=fake_rag)

    run_result = runner.run(dataset)
    assert run_result.results[0].retrieved_chunk_ids == [
        "chunk-alpha",
        "chunk-beta",
        "chunk-gamma",
    ]


def test_relevant_chunk_ids_preserved():
    """Verify relevant chunk IDs from the dataset case are preserved in result."""
    case = EvaluationCase(
        question="Question?",
        expected_answer="Answer.",
        relevant_chunk_ids=["gold-1", "gold-2"],
        case_type="factual",
    )
    dataset = EvaluationDataset(cases=[case])
    runner = EvaluationRunner(rag_service=FakeRAGService())

    run_result = runner.run(dataset)
    assert run_result.results[0].relevant_chunk_ids == ["gold-1", "gold-2"]


def test_case_type_preserved():
    """Verify case_type is preserved in the evaluation result."""
    case = EvaluationCase(
        question="Unanswerable query?",
        expected_answer="I don't know.",
        case_type="unanswerable",
    )
    dataset = EvaluationDataset(cases=[case])
    runner = EvaluationRunner(rag_service=FakeRAGService())

    run_result = runner.run(dataset)
    assert run_result.results[0].case_type == "unanswerable"


def test_top_k_forwarded_to_rag_service():
    """Verify top_k parameter is forwarded to RAGService for every case."""
    dataset = _build_sample_dataset()
    fake_rag = FakeRAGService()

    runner = EvaluationRunner(rag_service=fake_rag)

    # With integer top_k
    runner.run(dataset, top_k=5)
    assert len(fake_rag.calls) == 3
    assert all(call["top_k"] == 5 for call in fake_rag.calls)

    # With None top_k
    fake_rag.calls.clear()
    runner.run(dataset, top_k=None)
    assert len(fake_rag.calls) == 3
    assert all(call["top_k"] is None for call in fake_rag.calls)


def test_one_failed_case_does_not_stop_later_cases():
    """Verify an exception in one case does not abort the run and later cases execute."""
    dataset = EvaluationDataset(
        cases=[
            EvaluationCase(question="Q1", expected_answer="A1", case_type="factual"),
            EvaluationCase(question="Q2", expected_answer="A2", case_type="factual"),
            EvaluationCase(question="Q3", expected_answer="A3", case_type="factual"),
        ]
    )

    fake_rag = FakeRAGService(fail_questions={"Q2"})
    runner = EvaluationRunner(rag_service=fake_rag)

    run_result = runner.run(dataset)

    assert run_result.total_cases == 3
    assert run_result.successful_cases == 2
    assert run_result.failed_cases == 1

    # Case 1 succeeded
    assert run_result.results[0].question == "Q1"
    assert run_result.results[0].error is None
    assert run_result.results[0].actual_answer != ""

    # Case 2 failed
    failed_result = run_result.results[1]
    assert failed_result.question == "Q2"
    assert failed_result.expected_answer == "A2"
    assert failed_result.actual_answer == ""
    assert failed_result.retrieved_chunk_ids == []
    assert failed_result.error is not None
    assert "RAG failure for question: Q2" in failed_result.error

    # Case 3 succeeded
    assert run_result.results[2].question == "Q3"
    assert run_result.results[2].error is None
    assert run_result.results[2].actual_answer != ""


def test_empty_dataset_run():
    """Verify running an empty dataset produces zero counts and an empty results list."""
    dataset = EvaluationDataset(cases=[])
    runner = EvaluationRunner(rag_service=FakeRAGService())

    run_result = runner.run(dataset)

    assert run_result.total_cases == 0
    assert run_result.successful_cases == 0
    assert run_result.failed_cases == 0
    assert run_result.results == []


def test_default_rag_service_construction():
    """Verify EvaluationRunner instantiates real default RAGService safely without API calls."""
    runner = EvaluationRunner()
    assert isinstance(runner.rag_service, RAGService)


@pytest.mark.parametrize("invalid_dataset", [None, "invalid string", [1, 2, 3], {"cases": []}])
def test_invalid_dataset_input_rejected(invalid_dataset):
    """Verify non-EvaluationDataset input raises ValueError before running."""
    runner = EvaluationRunner(rag_service=FakeRAGService())

    with pytest.raises(ValueError, match="valid EvaluationDataset"):
        runner.run(invalid_dataset)
