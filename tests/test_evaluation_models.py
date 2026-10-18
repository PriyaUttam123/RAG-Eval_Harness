"""Unit tests for evaluation data models."""

import pytest
from pydantic import ValidationError

from backend.app.evaluation_models import EvaluationCase, EvaluationDataset


def test_valid_evaluation_case():
    """Verify creating a valid EvaluationCase with all fields populated."""
    case = EvaluationCase(
        question="What is the refund period?",
        expected_answer="Refunds are available within 30 days.",
        relevant_chunk_ids=["chunk-1", "chunk-2"],
        case_type="factual",
        metadata={"category": "returns", "difficulty": "easy"},
    )

    assert case.question == "What is the refund period?"
    assert case.expected_answer == "Refunds are available within 30 days."
    assert case.relevant_chunk_ids == ["chunk-1", "chunk-2"]
    assert case.case_type == "factual"
    assert case.metadata == {"category": "returns", "difficulty": "easy"}


def test_default_relevant_chunk_ids():
    """Verify relevant_chunk_ids defaults to an empty list."""
    case = EvaluationCase(
        question="What is the return policy?",
        expected_answer="30 days.",
        case_type="factual",
    )

    assert case.relevant_chunk_ids == []


def test_default_metadata():
    """Verify metadata defaults to an empty dictionary."""
    case = EvaluationCase(
        question="What is the return policy?",
        expected_answer="30 days.",
        case_type="factual",
    )

    assert case.metadata == {}


def test_valid_evaluation_dataset():
    """Verify creating an EvaluationDataset containing multiple cases."""
    case1 = EvaluationCase(
        question="Question 1?",
        expected_answer="Answer 1.",
        case_type="factual",
    )
    case2 = EvaluationCase(
        question="Question 2?",
        expected_answer="Answer 2.",
        case_type="unanswerable",
    )

    dataset = EvaluationDataset(cases=[case1, case2])

    assert len(dataset) == 2
    assert len(dataset.cases) == 2
    assert dataset[0].question == "Question 1?"
    assert dataset[1].case_type == "unanswerable"
    assert [c.question for c in dataset] == ["Question 1?", "Question 2?"]


@pytest.mark.parametrize("invalid_question", ["", "   ", " \n\t "])
def test_empty_question_rejected(invalid_question):
    """Verify empty or whitespace-only questions raise ValidationError."""
    with pytest.raises(ValidationError):
        EvaluationCase(
            question=invalid_question,
            expected_answer="Answer",
            case_type="factual",
        )


@pytest.mark.parametrize("invalid_case_type", ["", "   ", " \n\t "])
def test_empty_case_type_rejected(invalid_case_type):
    """Verify empty or whitespace-only case types raise ValidationError."""
    with pytest.raises(ValidationError):
        EvaluationCase(
            question="Valid question?",
            expected_answer="Answer",
            case_type=invalid_case_type,
        )


def test_missing_required_case_fields_rejected():
    """Verify omitting required fields from EvaluationCase raises ValidationError."""
    # Missing question
    with pytest.raises(ValidationError):
        EvaluationCase(
            expected_answer="Answer",
            case_type="factual",
        )

    # Missing expected_answer
    with pytest.raises(ValidationError):
        EvaluationCase(
            question="Question?",
            case_type="factual",
        )

    # Missing case_type
    with pytest.raises(ValidationError):
        EvaluationCase(
            question="Question?",
            expected_answer="Answer",
        )


def test_missing_cases_in_dataset_rejected():
    """Verify omitting 'cases' from EvaluationDataset raises ValidationError."""
    with pytest.raises(ValidationError):
        EvaluationDataset()
