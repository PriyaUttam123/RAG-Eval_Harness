"""Dataset manager for loading, validating, and separating golden and holdout benchmark sets."""

from pathlib import Path
from typing import Optional, Set, Union

try:
    from backend.app.evaluation_loader import EvaluationDatasetLoader
    from backend.app.evaluation_models import EvaluationCase, EvaluationDataset
except ImportError:
    from app.evaluation_loader import EvaluationDatasetLoader
    from app.evaluation_models import EvaluationCase, EvaluationDataset


def normalize_question(question: str) -> str:
    """Normalize question text by lowercasing and collapsing whitespace."""
    if not isinstance(question, str):
        return ""
    return " ".join(question.strip().lower().split())


class EvaluationDatasetManager:
    """Manages golden and holdout evaluation datasets with integrity validation."""

    def __init__(
        self,
        loader: Optional[EvaluationDatasetLoader] = None,
        golden_path: Optional[Union[str, Path]] = None,
        holdout_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize dataset manager with optional loader and path overrides.

        Args:
            loader: Optional EvaluationDatasetLoader instance.
            golden_path: Filepath to golden evaluation dataset JSON.
            holdout_path: Filepath to holdout evaluation dataset JSON.
        """
        self.loader = loader or EvaluationDatasetLoader()
        self.golden_path = Path(golden_path) if golden_path else Path("data/evaluation/golden.json")
        self.holdout_path = Path(holdout_path) if holdout_path else Path("data/evaluation/holdout.json")

    def load_golden(self, path: Optional[Union[str, Path]] = None) -> EvaluationDataset:
        """Load and validate the golden evaluation dataset.

        Args:
            path: Optional custom path to golden dataset JSON.

        Returns:
            Validated EvaluationDataset instance.
        """
        target = Path(path) if path else self.golden_path
        dataset = self.loader.load(target)
        self.validate_dataset(dataset)
        return dataset

    def load_holdout(self, path: Optional[Union[str, Path]] = None) -> EvaluationDataset:
        """Load and validate the holdout evaluation dataset.

        Args:
            path: Optional custom path to holdout dataset JSON.

        Returns:
            Validated EvaluationDataset instance.
        """
        target = Path(path) if path else self.holdout_path
        dataset = self.loader.load(target)
        self.validate_dataset(dataset)
        return dataset

    @classmethod
    def validate_dataset(cls, dataset: EvaluationDataset) -> None:
        """Validate dataset structure, fields, and question uniqueness.

        Args:
            dataset: EvaluationDataset instance to validate.

        Raises:
            ValueError: If dataset is invalid, fields are empty, or duplicate questions exist.
        """
        if dataset is None or not isinstance(dataset, EvaluationDataset):
            raise ValueError("dataset must be an instance of EvaluationDataset.")

        for index, case in enumerate(dataset.cases, start=1):
            if not isinstance(case.question, str) or not case.question.strip():
                raise ValueError(f"Case #{index} has an empty or invalid question.")

            if not isinstance(case.case_type, str) or not case.case_type.strip():
                raise ValueError(f"Case #{index} has an empty or invalid case_type.")

            if not isinstance(case.relevant_chunk_ids, list) or any(
                not isinstance(cid, str) for cid in case.relevant_chunk_ids
            ):
                raise ValueError(
                    f"Case #{index} relevant_chunk_ids must be a list of strings."
                )

        cls.validate_no_duplicate_questions(dataset)

    @classmethod
    def validate_no_duplicate_questions(cls, dataset: EvaluationDataset) -> None:
        """Ensure no duplicate questions exist within a single dataset.

        Args:
            dataset: EvaluationDataset to inspect.

        Raises:
            ValueError: If duplicate questions (ignoring case and whitespace) are found.
        """
        if dataset is None or not isinstance(dataset, EvaluationDataset):
            raise ValueError("dataset must be an instance of EvaluationDataset.")

        seen_questions: Set[str] = set()
        for case in dataset.cases:
            normalized = normalize_question(case.question)
            if normalized in seen_questions:
                raise ValueError(
                    f"Duplicate question detected within dataset: '{case.question}'"
                )
            seen_questions.add(normalized)

    @classmethod
    def validate_separation(
        cls,
        golden: EvaluationDataset,
        holdout: EvaluationDataset,
    ) -> None:
        """Ensure golden and holdout datasets have zero question overlap.

        Args:
            golden: Golden evaluation dataset used for tuning.
            holdout: Holdout evaluation dataset used for validation.

        Raises:
            ValueError: If overlapping questions are detected between datasets.
        """
        if golden is None or not isinstance(golden, EvaluationDataset):
            raise ValueError("golden dataset must be an instance of EvaluationDataset.")
        if holdout is None or not isinstance(holdout, EvaluationDataset):
            raise ValueError("holdout dataset must be an instance of EvaluationDataset.")

        cls.validate_dataset(golden)
        cls.validate_dataset(holdout)

        golden_questions = {normalize_question(c.question) for c in golden.cases}
        for case in holdout.cases:
            normalized = normalize_question(case.question)
            if normalized in golden_questions:
                raise ValueError(
                    f"Duplicate question across golden and holdout datasets: '{case.question}'"
                )


def validate_dataset(dataset: EvaluationDataset) -> None:
    """Validate dataset structure, fields, and internal uniqueness."""
    EvaluationDatasetManager.validate_dataset(dataset)


def validate_no_duplicate_questions(dataset: EvaluationDataset) -> None:
    """Ensure no duplicate questions exist within a single dataset."""
    EvaluationDatasetManager.validate_no_duplicate_questions(dataset)


def validate_separation(golden: EvaluationDataset, holdout: EvaluationDataset) -> None:
    """Ensure golden and holdout datasets have zero question overlap."""
    EvaluationDatasetManager.validate_separation(golden, holdout)
