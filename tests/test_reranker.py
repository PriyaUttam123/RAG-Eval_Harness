import pytest
from pydantic import ValidationError

from backend.app.reranker import RerankedResult, RerankerCandidate


def test_reranker_candidate_accepts_valid_data():
    candidate = RerankerCandidate(
        chunk_id="chunk-1",
        text="Python is a programming language.",
        metadata={"source_filename": "test.txt"},
        retrieval_score=0.85,
    )

    assert candidate.chunk_id == "chunk-1"
    assert candidate.text == "Python is a programming language."
    assert candidate.metadata["source_filename"] == "test.txt"
    assert candidate.retrieval_score == 0.85


def test_reranker_candidate_defaults_metadata_to_empty_dict():
    candidate = RerankerCandidate(
        chunk_id="chunk-1",
        text="Some text",
        retrieval_score=0.5,
    )

    assert candidate.metadata == {}


def test_reranked_result_accepts_valid_data():
    result = RerankedResult(
        chunk_id="chunk-1",
        text="Some text",
        metadata={"source_filename": "test.txt"},
        score=0.95,
        original_rank=2,
    )

    assert result.chunk_id == "chunk-1"
    assert result.score == 0.95
    assert result.original_rank == 2


def test_reranked_result_rejects_negative_original_rank():
    with pytest.raises(ValidationError):
        RerankedResult(
            chunk_id="chunk-1",
            text="Some text",
            score=0.95,
            original_rank=-1,
        )

from backend.app.reranker import (
    RerankedResult,
    RerankerCandidate,
    SimpleReranker,
)


@pytest.fixture
def reranker():
    return SimpleReranker()


def test_reranker_ranks_strong_lexical_match_higher(reranker):
    candidates = [
        RerankerCandidate(
            chunk_id="weak",
            text="Java is used for enterprise applications.",
            retrieval_score=0.9,
        ),
        RerankerCandidate(
            chunk_id="strong",
            text="Python is a programming language.",
            retrieval_score=0.5,
        ),
    ]

    results = reranker.rerank(
        query="Python programming language",
        candidates=candidates,
        top_k=2,
    )

    assert results[0].chunk_id == "strong"
    assert results[0].score == 1.0


def test_reranker_handles_partial_match(reranker):
    candidates = [
        RerankerCandidate(
            chunk_id="partial",
            text="Python is useful.",
            retrieval_score=0.5,
        ),
        RerankerCandidate(
            chunk_id="none",
            text="Database administration.",
            retrieval_score=0.8,
        ),
    ]

    results = reranker.rerank(
        query="Python programming language",
        candidates=candidates,
        top_k=2,
    )

    assert results[0].chunk_id == "partial"
    assert results[0].score > results[1].score


def test_reranker_handles_no_overlap(reranker):
    candidates = [
        RerankerCandidate(
            chunk_id="chunk-1",
            text="Database administration.",
            retrieval_score=0.5,
        ),
    ]

    results = reranker.rerank(
        query="Python programming",
        candidates=candidates,
        top_k=1,
    )

    assert results[0].score == 0.0


def test_reranker_returns_empty_for_empty_candidates(reranker):
    results = reranker.rerank(
        query="Python",
        candidates=[],
        top_k=5,
    )

    assert results == []


def test_reranker_rejects_empty_query(reranker):
    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank(
            query="   ",
            candidates=[],
            top_k=5,
        )


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_reranker_rejects_invalid_top_k(reranker, top_k):
    with pytest.raises(ValueError, match="positive integer"):
        reranker.rerank(
            query="Python",
            candidates=[],
            top_k=top_k,
        )


def test_reranker_respects_top_k(reranker):
    candidates = [
        RerankerCandidate(
            chunk_id=f"chunk-{index}",
            text="Python programming language",
            retrieval_score=0.5,
        )
        for index in range(5)
    ]

    results = reranker.rerank(
        query="Python",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2


def test_reranker_preserves_metadata(reranker):
    candidates = [
        RerankerCandidate(
            chunk_id="chunk-1",
            text="Python programming",
            metadata={"source_filename": "guide.txt"},
            retrieval_score=0.5,
        )
    ]

    results = reranker.rerank(
        query="Python",
        candidates=candidates,
        top_k=1,
    )

    assert results[0].metadata == {"source_filename": "guide.txt"}


def test_reranker_preserves_chunk_id(reranker):
    candidates = [
        RerankerCandidate(
            chunk_id="important-id",
            text="Python programming",
            retrieval_score=0.5,
        )
    ]

    results = reranker.rerank(
        query="Python",
        candidates=candidates,
        top_k=1,
    )

    assert results[0].chunk_id == "important-id"


def test_reranker_uses_original_rank_as_tie_breaker(reranker):
    candidates = [
        RerankerCandidate(
            chunk_id="first",
            text="Python",
            retrieval_score=0.5,
        ),
        RerankerCandidate(
            chunk_id="second",
            text="Python",
            retrieval_score=0.9,
        ),
    ]

    results = reranker.rerank(
        query="Python",
        candidates=candidates,
        top_k=2,
    )

    assert results[0].chunk_id == "first"
    assert results[1].chunk_id == "second"
    assert results[0].original_rank == 0
    assert results[1].original_rank == 1


def test_reranker_is_deterministic(reranker):
    candidates = [
        RerankerCandidate(
            chunk_id="a",
            text="Python programming",
            retrieval_score=0.5,
        ),
        RerankerCandidate(
            chunk_id="b",
            text="Python programming",
            retrieval_score=0.4,
        ),
    ]

    first = reranker.rerank("Python", candidates, 2)
    second = reranker.rerank("Python", candidates, 2)

    assert first == second        