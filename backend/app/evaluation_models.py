"""Evaluation data models for RAG benchmark datasets."""

from typing import Any, List

from pydantic import BaseModel, Field, field_validator


class EvaluationCase(BaseModel):
    """A single evaluation benchmark case."""

    question: str = Field(..., description="The query to evaluate.")
    expected_answer: str = Field(..., description="The ground truth or expected answer.")
    relevant_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Ground-truth relevant chunk IDs for retrieval evaluation.",
    )
    case_type: str = Field(
        ...,
        description="Category of the evaluation case (e.g., 'factual', 'unanswerable').",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with this case.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("question must be a non-empty string.")
        return value.strip()

    @field_validator("case_type")
    @classmethod
    def validate_case_type(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("case_type must be a non-empty string.")
        return value.strip()


class EvaluationDataset(BaseModel):
    """Collection of evaluation benchmark cases."""

    cases: List[EvaluationCase] = Field(
        ...,
        description="List of evaluation cases in the dataset.",
    )

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, index: int) -> EvaluationCase:
        return self.cases[index]
