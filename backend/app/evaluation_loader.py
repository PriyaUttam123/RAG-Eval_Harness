"""Loader service for parsing and validating evaluation datasets from JSON."""

import json
from pathlib import Path
from typing import Optional, Union

try:
    from backend.app.evaluation_models import EvaluationDataset
except ImportError:
    from app.evaluation_models import EvaluationDataset


class EvaluationDatasetLoader:
    """Loads and validates evaluation datasets from JSON files."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize loader with an optional default file path.

        Args:
            file_path: Optional default path to the evaluation dataset JSON file.
        """
        self.file_path = Path(file_path) if file_path is not None else None

    def load(self, file_path: Optional[Union[str, Path]] = None) -> EvaluationDataset:
        """Load and validate an EvaluationDataset from a JSON file.

        Args:
            file_path: Path to the JSON dataset file. If not provided, uses
                the path supplied during initialization.

        Returns:
            Validated EvaluationDataset instance.

        Raises:
            ValueError: If no file path is provided, or JSON format/structure is invalid.
            FileNotFoundError: If the specified file path does not exist.
            ValidationError: If the dataset content fails Pydantic validation.
        """
        target = file_path if file_path is not None else self.file_path
        if target is None:
            raise ValueError("A file path must be provided to load the dataset.")

        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

        try:
            raw_text = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise ValueError(f"Could not read file '{path}': {exc}") from exc

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON format in evaluation dataset: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected top-level JSON object in evaluation dataset, got {type(data).__name__}."
            )

        if "cases" not in data:
            raise ValueError("Missing required 'cases' field in evaluation dataset.")

        return EvaluationDataset.model_validate(data)
