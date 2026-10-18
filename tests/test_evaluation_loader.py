"""Unit tests for EvaluationDatasetLoader."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.evaluation_loader import EvaluationDatasetLoader
from backend.app.evaluation_models import EvaluationDataset


def test_load_valid_single_case_dataset(tmp_path: Path):
    """Verify loading a valid JSON dataset with a single evaluation case."""
    dataset_file = tmp_path / "single_case.json"
    dataset_data = {
        "cases": [
            {
                "question": "What is the policy?",
                "expected_answer": "Refund within 30 days.",
                "relevant_chunk_ids": ["chunk-1"],
                "case_type": "factual",
                "metadata": {"section": "returns"},
            }
        ]
    }
    dataset_file.write_text(json.dumps(dataset_data), encoding="utf-8")

    loader = EvaluationDatasetLoader()
    dataset = loader.load(dataset_file)

    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset) == 1
    assert dataset.cases[0].question == "What is the policy?"
    assert dataset.cases[0].expected_answer == "Refund within 30 days."
    assert dataset.cases[0].relevant_chunk_ids == ["chunk-1"]
    assert dataset.cases[0].case_type == "factual"
    assert dataset.cases[0].metadata == {"section": "returns"}


def test_load_multiple_evaluation_cases(tmp_path: Path):
    """Verify loading a JSON dataset containing multiple evaluation cases."""
    dataset_file = tmp_path / "multiple_cases.json"
    dataset_data = {
        "cases": [
            {
                "question": "Question 1?",
                "expected_answer": "Answer 1",
                "relevant_chunk_ids": ["c1", "c2"],
                "case_type": "factual",
            },
            {
                "question": "Question 2?",
                "expected_answer": "Answer 2",
                "relevant_chunk_ids": [],
                "case_type": "unanswerable",
            },
            {
                "question": "Question 3?",
                "expected_answer": "Answer 3",
                "case_type": "hypothetical",
                "metadata": {"level": "hard"},
            },
        ]
    }
    dataset_file.write_text(json.dumps(dataset_data), encoding="utf-8")

    loader = EvaluationDatasetLoader()
    dataset = loader.load(dataset_file)

    assert len(dataset) == 3
    assert [c.case_type for c in dataset] == ["factual", "unanswerable", "hypothetical"]
    assert dataset.cases[2].relevant_chunk_ids == []
    assert dataset.cases[2].metadata == {"level": "hard"}


def test_load_missing_file_raises_error(tmp_path: Path):
    """Verify attempting to load a non-existent file raises FileNotFoundError."""
    missing_file = tmp_path / "non_existent.json"
    loader = EvaluationDatasetLoader()

    with pytest.raises(FileNotFoundError, match="Evaluation dataset file not found"):
        loader.load(missing_file)


def test_load_invalid_json_raises_error(tmp_path: Path):
    """Verify attempting to load malformed JSON raises ValueError."""
    corrupted_file = tmp_path / "corrupted.json"
    corrupted_file.write_text("{ invalid json: [", encoding="utf-8")

    loader = EvaluationDatasetLoader()

    with pytest.raises(ValueError, match="Invalid JSON format"):
        loader.load(corrupted_file)


@pytest.mark.parametrize(
    "invalid_content",
    [
        json.dumps(["just", "a", "list"]),
        json.dumps("a raw string"),
        json.dumps(12345),
        json.dumps({"not_cases": []}),
    ],
)
def test_load_invalid_dataset_structure(tmp_path: Path, invalid_content: str):
    """Verify non-dict JSON or JSON missing 'cases' raises ValueError."""
    invalid_file = tmp_path / "invalid_structure.json"
    invalid_file.write_text(invalid_content, encoding="utf-8")

    loader = EvaluationDatasetLoader()

    with pytest.raises(ValueError):
        loader.load(invalid_file)


def test_load_validation_failure(tmp_path: Path):
    """Verify cases that violate schema constraints raise ValidationError."""
    invalid_file = tmp_path / "invalid_case.json"
    invalid_file.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question": "   ",  # invalid whitespace-only question
                        "expected_answer": "Answer",
                        "case_type": "factual",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loader = EvaluationDatasetLoader()

    with pytest.raises(ValidationError):
        loader.load(invalid_file)


def test_loader_uses_constructor_path(tmp_path: Path):
    """Verify loader can load from the path provided at initialization."""
    dataset_file = tmp_path / "init_path.json"
    dataset_file.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question": "Init path question?",
                        "expected_answer": "Answer",
                        "case_type": "factual",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loader = EvaluationDatasetLoader(file_path=dataset_file)
    dataset = loader.load()

    assert len(dataset) == 1
    assert dataset.cases[0].question == "Init path question?"


def test_loader_missing_path_raises_value_error():
    """Verify calling load() without constructor or argument path raises ValueError."""
    loader = EvaluationDatasetLoader()

    with pytest.raises(ValueError, match="A file path must be provided"):
        loader.load()
