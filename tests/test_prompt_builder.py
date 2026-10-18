import pytest

from backend.app.prompt_builder import PromptContext, RAGPromptBuilder


def test_build_creates_system_and_user_messages():
    builder = RAGPromptBuilder()

    messages = builder.build(
        question="What is the refund period?",
        contexts=[
            PromptContext(
                text="Customers may request a refund within 30 days.",
                source_filename="policy.txt",
                chunk_index=0,
            )
        ],
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    assert "only the provided context" in messages[0]["content"]
    assert "refund period" in messages[1]["content"]
    assert "30 days" in messages[1]["content"]
    assert "policy.txt" in messages[1]["content"]


def test_build_strips_question_whitespace():
    builder = RAGPromptBuilder()

    messages = builder.build(
        question="  What is the refund period?  ",
        contexts=[],
    )

    assert "Question:\nWhat is the refund period?" in messages[1]["content"]


def test_build_handles_empty_context():
    builder = RAGPromptBuilder()

    messages = builder.build(
        question="What is the refund period?",
        contexts=[],
    )

    assert "[No relevant context was retrieved.]" in messages[1]["content"]


def test_build_rejects_empty_question():
    builder = RAGPromptBuilder()

    with pytest.raises(ValueError):
        builder.build("", [])


def test_build_includes_multiple_contexts():
    builder = RAGPromptBuilder()

    messages = builder.build(
        question="What are the refund rules?",
        contexts=[
            PromptContext(
                text="Refunds are available within 30 days.",
                source_filename="policy.txt",
                chunk_index=0,
            ),
            PromptContext(
                text="Refund requests require proof of purchase.",
                source_filename="returns.txt",
                chunk_index=2,
            ),
        ],
    )

    user_content = messages[1]["content"]

    assert "[Context 1 | Source: policy.txt]" in user_content
    assert "[Context 2 | Source: returns.txt]" in user_content
    assert "Refunds are available within 30 days." in user_content
    assert "proof of purchase" in user_content