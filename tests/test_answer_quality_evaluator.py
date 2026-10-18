"""Unit tests for AnswerQualityEvaluator, AnswerQualityScore, and SimpleAnswerQualityJudge."""

from typing import List

import pytest
from pydantic import ValidationError

from backend.app.answer_quality_evaluator import (
    AnswerQualityEvaluator,
    AnswerQualityJudge,
    AnswerQualityScore,
    SimpleAnswerQualityJudge,
)


# ============================================================================
# AnswerQualityScore validation tests
# ============================================================================


def test_answer_quality_score_valid_values():
    """Verify valid boundary and intermediate score values are accepted."""
    score = AnswerQualityScore(
        correctness=1.0,
        groundedness=0.5,
        completeness=0.8,
        overall=0.7667,
        explanation="Valid score explanation.",
    )

    assert score.correctness == 1.0
    assert score.groundedness == 0.5
    assert score.completeness == 0.8
    assert score.overall == 0.7667
    assert score.explanation == "Valid score explanation."


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"correctness": -0.01, "groundedness": 0.5, "completeness": 0.5, "overall": 0.5},
        {"correctness": 0.5, "groundedness": -0.1, "completeness": 0.5, "overall": 0.5},
        {"correctness": 0.5, "groundedness": 0.5, "completeness": -0.5, "overall": 0.5},
        {"correctness": 0.5, "groundedness": 0.5, "completeness": 0.5, "overall": -0.001},
    ],
)
def test_answer_quality_score_rejects_values_below_zero(invalid_kwargs):
    """Verify any metric value below 0.0 raises ValidationError."""
    with pytest.raises(ValidationError):
        AnswerQualityScore(**invalid_kwargs)


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"correctness": 1.01, "groundedness": 0.5, "completeness": 0.5, "overall": 0.5},
        {"correctness": 0.5, "groundedness": 1.1, "completeness": 0.5, "overall": 0.5},
        {"correctness": 0.5, "groundedness": 0.5, "completeness": 2.0, "overall": 0.5},
        {"correctness": 0.5, "groundedness": 0.5, "completeness": 0.5, "overall": 1.001},
    ],
)
def test_answer_quality_score_rejects_values_above_one(invalid_kwargs):
    """Verify any metric value above 1.0 raises ValidationError."""
    with pytest.raises(ValidationError):
        AnswerQualityScore(**invalid_kwargs)


# ============================================================================
# Injected judge tests
# ============================================================================


class FakeAnswerJudge:
    def __init__(self, returned_score: AnswerQualityScore):
        self.returned_score = returned_score
        self.received_call = None

    def judge(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        context: List[str],
    ) -> AnswerQualityScore:
        self.received_call = {
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "context": context,
        }
        return self.returned_score


def test_evaluator_delegates_to_injected_judge():
    """Verify evaluator delegates judgment to the injected judge and preserves returned score."""
    expected_score = AnswerQualityScore(
        correctness=0.9,
        groundedness=0.85,
        completeness=0.95,
        overall=0.9,
        explanation="Fake judge score.",
    )
    fake_judge = FakeAnswerJudge(returned_score=expected_score)
    evaluator = AnswerQualityEvaluator(judge=fake_judge)

    result = evaluator.evaluate_case(
        question="What is the refund period?",
        expected_answer="30 days.",
        actual_answer="Customers have 30 days to request a refund.",
        context=["Refund policy text snippet."],
    )

    assert result == expected_score
    assert fake_judge.received_call["question"] == "What is the refund period?"
    assert fake_judge.received_call["expected_answer"] == "30 days."
    assert fake_judge.received_call["actual_answer"] == "Customers have 30 days to request a refund."
    assert fake_judge.received_call["context"] == ["Refund policy text snippet."]


def test_evaluator_defaults_to_simple_judge():
    """Verify evaluator defaults to SimpleAnswerQualityJudge when judge is not supplied."""
    evaluator = AnswerQualityEvaluator()
    assert isinstance(evaluator.judge, SimpleAnswerQualityJudge)


# ============================================================================
# SimpleAnswerQualityJudge tests
# ============================================================================


def test_simple_judge_exact_answer_gets_perfect_correctness_and_completeness():
    """Verify identical expected and actual answers achieve 1.0 correctness and completeness."""
    judge = SimpleAnswerQualityJudge()
    score = judge.judge(
        question="What is the capital?",
        expected_answer="Paris is the capital of France.",
        actual_answer="paris is the capital of france.",
        context=["Paris is the capital of France."],
    )

    assert score.correctness == pytest.approx(1.0)
    assert score.completeness == pytest.approx(1.0)
    assert score.groundedness == pytest.approx(1.0)
    assert score.overall == pytest.approx(1.0)


def test_simple_judge_empty_answer_gets_zero_scores():
    """Verify empty actual answer receives 0.0 across all dimensions."""
    judge = SimpleAnswerQualityJudge()
    score = judge.judge(
        question="What is the capital?",
        expected_answer="Paris.",
        actual_answer="",
        context=["Paris is the capital."],
    )

    assert score.correctness == 0.0
    assert score.completeness == 0.0
    assert score.groundedness == 0.0
    assert score.overall == 0.0


def test_simple_judge_partial_token_overlap_scores():
    """Verify partial token overlap results in scores between 0.0 and 1.0."""
    judge = SimpleAnswerQualityJudge()
    score = judge.judge(
        question="Tell me about Paris.",
        expected_answer="Paris is the capital city of France.",
        actual_answer="Paris is a beautiful city in Europe.",
        context=["Paris is located in Europe."],
    )

    assert 0.0 < score.correctness < 1.0
    assert 0.0 < score.completeness < 1.0
    assert 0.0 < score.overall < 1.0


def test_simple_judge_matching_context_gives_non_zero_groundedness():
    """Verify actual answer with overlapping context produces positive groundedness."""
    judge = SimpleAnswerQualityJudge()
    score = judge.judge(
        question="What is the return period?",
        expected_answer="30 days.",
        actual_answer="The return period is 30 days.",
        context=["The return period is 30 days from delivery."],
    )

    assert score.groundedness > 0.0
    assert score.groundedness == pytest.approx(1.0)


def test_simple_judge_unrelated_context_produces_zero_groundedness():
    """Verify actual answer with no token overlap in context produces 0.0 groundedness."""
    judge = SimpleAnswerQualityJudge()
    score = judge.judge(
        question="What is the speed of light?",
        expected_answer="300000 km per second",
        actual_answer="Fast light photons",
        context=["Apples and bananas grow in tropical orchards."],
    )

    assert score.groundedness == pytest.approx(0.0)


def test_simple_judge_overall_is_arithmetic_mean():
    """Verify overall score is strictly the arithmetic mean of correctness, groundedness, completeness."""
    judge = SimpleAnswerQualityJudge()
    score = judge.judge(
        question="Question?",
        expected_answer="First second third fourth",
        actual_answer="First second fifth sixth",
        context=["First fifth seventh"],
    )

    expected_overall = (score.correctness + score.groundedness + score.completeness) / 3.0
    assert score.overall == pytest.approx(expected_overall)


def test_simple_judge_explanation_identifies_deterministic_heuristic():
    """Verify explanation notes the score is from a deterministic heuristic rather than an LLM."""
    judge = SimpleAnswerQualityJudge()
    score = judge.judge(
        question="Query?",
        expected_answer="Expected",
        actual_answer="Actual",
        context=["Context"],
    )

    explanation_lower = score.explanation.lower()
    assert "deterministic heuristic" in explanation_lower
    assert "not an llm" in explanation_lower


# ============================================================================
# Input validation tests
# ============================================================================


@pytest.mark.parametrize("invalid_question", ["", "   ", " \n\t "])
def test_evaluator_rejects_empty_question(invalid_question):
    """Verify empty or whitespace-only question raises ValueError."""
    evaluator = AnswerQualityEvaluator()

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        evaluator.evaluate_case(
            question=invalid_question,
            expected_answer="Expected",
            actual_answer="Actual",
            context=["Context"],
        )


@pytest.mark.parametrize("invalid_question_type", [None, 123, ["question"], {"q": "text"}])
def test_evaluator_rejects_invalid_question_type(invalid_question_type):
    """Verify non-string question raises TypeError."""
    evaluator = AnswerQualityEvaluator()

    with pytest.raises(TypeError, match="question must be a string"):
        evaluator.evaluate_case(
            question=invalid_question_type,
            expected_answer="Expected",
            actual_answer="Actual",
            context=["Context"],
        )


@pytest.mark.parametrize("invalid_expected_type", [None, 123, ["expected"], False])
def test_evaluator_rejects_invalid_expected_answer_type(invalid_expected_type):
    """Verify non-string expected_answer raises TypeError."""
    evaluator = AnswerQualityEvaluator()

    with pytest.raises(TypeError, match="expected_answer must be a string"):
        evaluator.evaluate_case(
            question="Valid question?",
            expected_answer=invalid_expected_type,
            actual_answer="Actual",
            context=["Context"],
        )


@pytest.mark.parametrize("invalid_actual_type", [None, 123, ["actual"], True])
def test_evaluator_rejects_invalid_actual_answer_type(invalid_actual_type):
    """Verify non-string actual_answer raises TypeError."""
    evaluator = AnswerQualityEvaluator()

    with pytest.raises(TypeError, match="actual_answer must be a string"):
        evaluator.evaluate_case(
            question="Valid question?",
            expected_answer="Expected",
            actual_answer=invalid_actual_type,
            context=["Context"],
        )


@pytest.mark.parametrize(
    "invalid_context",
    [
        "not a list",
        None,
        123,
        [1, 2, 3],
        ["valid", None],
    ],
)
def test_evaluator_rejects_invalid_context_type(invalid_context):
    """Verify non-list context or list containing non-strings raises TypeError."""
    evaluator = AnswerQualityEvaluator()

    with pytest.raises(TypeError, match="context must be a list of strings"):
        evaluator.evaluate_case(
            question="Valid question?",
            expected_answer="Expected",
            actual_answer="Actual",
            context=invalid_context,
        )
