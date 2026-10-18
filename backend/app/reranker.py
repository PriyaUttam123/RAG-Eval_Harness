from typing import Any, Dict, List, Protocol

from pydantic import BaseModel, Field


class RerankerCandidate(BaseModel):
    """A candidate chunk produced by an upstream retrieval strategy."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieval_score: float


class RerankedResult(BaseModel):
    """A candidate after reranking."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float
    original_rank: int = Field(ge=0)


class Reranker(Protocol):
    """Interface implemented by reranking strategies."""

    def rerank(
        self,
        query: str,
        candidates: List[RerankerCandidate],
        top_k: int,
    ) -> List[RerankedResult]:
        """Rerank retrieval candidates and return the best results."""
        ...


class SimpleReranker:
    """Deterministic lexical reranker used as a baseline."""

    def rerank(
        self,
        query: str,
        candidates: List[RerankerCandidate],
        top_k: int,
    ) -> List[RerankedResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if not candidates:
            return []

        query_tokens = self._tokenize(query)

        scored = []

        for original_rank, candidate in enumerate(candidates):
            candidate_tokens = self._tokenize(candidate.text)

            if query_tokens:
                overlap = query_tokens.intersection(candidate_tokens)
                score = len(overlap) / len(query_tokens)
            else:
                score = 0.0

            scored.append(
                RerankedResult(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    metadata=candidate.metadata,
                    score=score,
                    original_rank=original_rank,
                )
            )

        scored.sort(
            key=lambda result: (
                -result.score,
                result.original_rank,
            )
        )

        return scored[:top_k]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Normalize text into lowercase alphanumeric tokens."""
        import re

        return set(re.findall(r"\b\w+\b", text.lower()))