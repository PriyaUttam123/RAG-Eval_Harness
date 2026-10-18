"""Provider-independent answer-quality evaluation interface and baseline judge."""

import re
from typing import List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class AnswerQualityScore(BaseModel):
    """Quality scores across evaluation dimensions."""

    correctness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Factual correctness of the actual answer relative to ground truth.",
    )
    groundedness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extent to which the answer is grounded in retrieved context snippets.",
    )
    completeness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Completeness of coverage of the expected answer information.",
    )
    overall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Composite overall answer quality score.",
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation or reasoning for assigned scores.",
    )


@runtime_checkable
class AnswerQualityJudge(Protocol):
    """Protocol for scoring question-answering output quality."""

    def judge(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        context: List[str],
    ) -> AnswerQualityScore:
        """Score answer quality across correctness, groundedness, and completeness."""
        ...


class SimpleAnswerQualityJudge:
    """Deterministic token-overlap heuristic judge for baseline testing without an LLM.

    NOTE: This is a lexical heuristic baseline for testing and pipeline validation.
    It does not represent semantic understanding or human/LLM-level evaluation quality.
    """

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Normalize case and extract alphanumeric word tokens."""
        return re.findall(r"\w+", text.lower())

    def judge(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        context: List[str],
    ) -> AnswerQualityScore:
        """Compute deterministic token-overlap scores."""
        norm_expected = expected_answer.strip().lower()
        norm_actual = actual_answer.strip().lower()

        # Empty actual answer gets zero scores
        if not norm_actual:
            return AnswerQualityScore(
                correctness=0.0,
                groundedness=0.0,
                completeness=0.0,
                overall=0.0,
                explanation="Deterministic heuristic baseline: actual answer is empty.",
            )

        actual_tokens = set(self._tokenize(actual_answer))
        expected_tokens = set(self._tokenize(expected_answer))
        context_tokens = set(self._tokenize(" ".join(context))) if context else set()

        # Exact normalized match gets perfect correctness and completeness
        if norm_expected == norm_actual:
            correctness = 1.0
            completeness = 1.0
        else:
            if not expected_tokens:
                correctness = 1.0 if not actual_tokens else 0.0
                completeness = 1.0
            else:
                overlap = actual_tokens.intersection(expected_tokens)
                correctness = len(overlap) / len(actual_tokens) if actual_tokens else 0.0
                completeness = len(overlap) / len(expected_tokens) if expected_tokens else 0.0

        # Groundedness: actual answer token overlap with retrieved context
        if not context_tokens or not actual_tokens:
            groundedness = 0.0
        else:
            grounded_overlap = actual_tokens.intersection(context_tokens)
            groundedness = len(grounded_overlap) / len(actual_tokens)

        # Overall: arithmetic mean of the three dimensions
        overall = (correctness + groundedness + completeness) / 3.0

        explanation = (
            "Deterministic heuristic baseline: scores computed using token overlap "
            "(precision, recall, and context grounding). Not an LLM or human judgment."
        )

        return AnswerQualityScore(
            correctness=float(correctness),
            groundedness=float(groundedness),
            completeness=float(completeness),
            overall=float(overall),
            explanation=explanation,
        )


class AnswerQualityEvaluator:
    """Evaluates answer quality by delegating scoring to an AnswerQualityJudge."""

    def __init__(self, judge: Optional[AnswerQualityJudge] = None) -> None:
        """Initialize evaluator with an optional AnswerQualityJudge.

        Args:
            judge: Optional AnswerQualityJudge instance. If not provided,
                defaults to SimpleAnswerQualityJudge baseline.
        """
        self.judge = judge or SimpleAnswerQualityJudge()

    def evaluate_case(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        context: List[str],
    ) -> AnswerQualityScore:
        """Evaluate a single question-answering case.

        Args:
            question: The user query string.
            expected_answer: Ground truth target answer.
            actual_answer: Model-generated answer.
            context: List of retrieved context text snippets.

        Returns:
            AnswerQualityScore containing correctness, groundedness, completeness, and overall score.

        Raises:
            TypeError: If input types are invalid.
            ValueError: If question is empty or whitespace-only.
        """
        if not isinstance(question, str):
            raise TypeError("question must be a string.")
        if not question.strip():
            raise ValueError("question must be a non-empty string.")
        if not isinstance(expected_answer, str):
            raise TypeError("expected_answer must be a string.")
        if not isinstance(actual_answer, str):
            raise TypeError("actual_answer must be a string.")
        if not isinstance(context, list) or any(not isinstance(c, str) for c in context):
            raise TypeError("context must be a list of strings.")

        return self.judge.judge(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            context=context,
        )
