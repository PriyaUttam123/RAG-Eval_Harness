"""Unit tests for EvaluationDatasetManager and dataset validation."""

import json
from pathlib import Path

import pytest

from backend.app.evaluation_dataset_manager import (
    EvaluationDatasetManager,
    normalize_question,
    validate_dataset,
    validate_no_duplicate_questions,
    validate_separation,
)
from backend.app.evaluation_models import EvaluationCase, EvaluationDataset


def test_load_golden_and_holdout_default_empty_files():
    """Verify loading default initialized data/evaluation/golden.json and holdout.json."""
    manager = EvaluationDatasetManager()

    golden = manager.load_golden()
    assert isinstance(golden, EvaluationDataset)
    assert len(golden) == 0

    holdout = manager.load_holdout()
    assert isinstance(holdout, EvaluationDataset)
    assert len(holdout) == 0

    # Ensure empty datasets pass separation validation
    manager.validate_separation(golden, holdout)


def test_load_golden_and_holdout_custom_paths(tmp_path: Path):
    """Verify loading from custom filepaths."""
    golden_path = tmp_path / "custom_golden.json"
    holdout_path = tmp_path / "custom_holdout.json"

    golden_data = {
        "cases": [
            {
                "question": "Golden question 1?",
                "expected_answer": "Answer 1",
                "relevant_chunk_ids": ["c1"],
                "case_type": "factual",
            }
        ]
    }
    holdout_data = {
        "cases": [
            {
                "question": "Holdout question 1?",
                "expected_answer": "Answer 2",
                "relevant_chunk_ids": ["c2"],
                "case_type": "factual",
            }
        ]
    }

    golden_path.write_text(json.dumps(golden_data), encoding="utf-8")
    holdout_path.write_text(json.dumps(holdout_data), encoding="utf-8")

    manager = EvaluationDatasetManager(
        golden_path=golden_path,
        holdout_path=holdout_path,
    )

    golden = manager.load_golden()
    assert len(golden) == 1
    assert golden.cases[0].question == "Golden question 1?"

    holdout = manager.load_holdout()
    assert len(holdout) == 1
    assert holdout.cases[0].question == "Holdout question 1?"

    manager.validate_separation(golden, holdout)


def test_duplicate_questions_within_single_dataset_rejected():
    """Verify duplicate questions within a single dataset raise ValueError."""
    case1 = EvaluationCase(question="What is the refund period?", expected_answer="A1", case_type="factual")
    case2 = EvaluationCase(question="What is the refund period?", expected_answer="A2", case_type="factual")

    dataset = EvaluationDataset(cases=[case1, case2])

    with pytest.raises(ValueError, match="Duplicate question detected"):
        validate_no_duplicate_questions(dataset)


def test_normalized_duplicate_detection_case_and_whitespace():
    """Verify duplicate questions with different case and spacing are detected and rejected."""
    case1 = EvaluationCase(question="  What is   the Refund Period?  ", expected_answer="A1", case_type="factual")
    case2 = EvaluationCase(question="what is the refund period?", expected_answer="A2", case_type="factual")

    dataset = EvaluationDataset(cases=[case1, case2])

    with pytest.raises(ValueError, match="Duplicate question detected"):
        validate_dataset(dataset)


def test_duplicate_question_across_golden_and_holdout_rejected():
    """Verify overlapping questions between golden and holdout datasets trigger validation error."""
    golden = EvaluationDataset(
        cases=[
            EvaluationCase(question="How to return an item?", expected_answer="A1", case_type="factual"),
            EvaluationCase(question="Where is my tracking number?", expected_answer="A2", case_type="factual"),
        ]
    )

    # Holdout shares the same normalized question with golden
    holdout = EvaluationDataset(
        cases=[
            EvaluationCase(question="HOW TO RETURN AN ITEM?", expected_answer="A1", case_type="factual"),
            EvaluationCase(question="What are your hours?", expected_answer="A3", case_type="factual"),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate question across golden and holdout"):
        validate_separation(golden, holdout)


@pytest.mark.parametrize("invalid_dataset", [None, "dataset", [1, 2, 3], {"cases": []}])
def test_invalid_dataset_object_rejected(invalid_dataset):
    """Verify passing non-EvaluationDataset objects raises ValueError."""
    with pytest.raises(ValueError, match="must be an instance of EvaluationDataset"):
        validate_dataset(invalid_dataset)

    with pytest.raises(ValueError, match="must be an instance of EvaluationDataset"):
        validate_no_duplicate_questions(invalid_dataset)


def test_empty_case_type_rejected():
    """Verify cases with empty or whitespace-only case_type are rejected."""
    # Constructing via dict or modified instance
    case = EvaluationCase(
        question="Valid question?",
        expected_answer="Answer",
        case_type="factual",
    )
    # Simulate a bypass or mutation
    case.case_type = "   "
    dataset = EvaluationDataset(cases=[case])

    with pytest.raises(ValueError, match="empty or invalid case_type"):
        validate_dataset(dataset)


def test_unanswerable_case_with_empty_relevant_chunk_ids_allowed():
    """Verify unanswerable cases with empty relevant_chunk_ids are valid."""
    case = EvaluationCase(
        question="What is the color of magic?",
        expected_answer="Not found in documentation.",
        relevant_chunk_ids=[],
        case_type="unanswerable",
    )
    dataset = EvaluationDataset(cases=[case])

    # Should not raise
    validate_dataset(dataset)


def test_relevant_chunk_ids_must_contain_strings():
    """Verify relevant_chunk_ids containing non-string items is rejected."""
    case = EvaluationCase(
        question="Valid question?",
        expected_answer="Answer",
        relevant_chunk_ids=["chunk-1"],
        case_type="factual",
    )
    # Simulate non-string element in list
    case.relevant_chunk_ids = ["chunk-1", 123]  # type: ignore
    dataset = EvaluationDataset(cases=[case])

    with pytest.raises(ValueError, match="relevant_chunk_ids must be a list of strings"):
        validate_dataset(dataset)


def test_normalize_question_behavior():
    """Verify normalization handles mixed casing and extraneous whitespace."""
    raw = "  \n  What   is   RAG? \t "
    assert normalize_question(raw) == "what is rag?"
    assert normalize_question(None) == ""
