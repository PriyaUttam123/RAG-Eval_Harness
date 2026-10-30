"""Structured failure-analysis models for evaluation results."""

from typing import List, Optional

from pydantic import BaseModel, Field


class EvaluationFailure(BaseModel):
    """Details of one failure found during evaluation."""

    case_id: str
    question: str
    expected_answer: str = ""
    actual_answer: str = ""
    failure_type: str
    error_message: Optional[str] = None
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    answer_quality_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )


class FailureAnalysis(BaseModel):
    """Collection of failures found in an evaluation run."""

    failures: List[EvaluationFailure] = Field(default_factory=list)
    total_failures: int = Field(..., ge=0)
