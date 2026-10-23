"""RAG orchestration service coordinating retrieval and generation."""

from typing import Any, List, Optional

from pydantic import BaseModel

try:
    from backend.app.generation_service import GenerationService
    from backend.app.prompt_builder import PromptContext
    from backend.app.retrieval_service import RetrievalService, RetrievedChunk
except ImportError:
    from app.generation_service import GenerationService
    from app.prompt_builder import PromptContext
    from app.retrieval_service import RetrievalService, RetrievedChunk


class RAGSource(BaseModel):
    """Information about a retrieved source chunk used for grounding."""

    chunk_id: str
    text: str
    source_filename: str = ""
    chunk_index: int = -1
    distance: float


class RAGAnswer(BaseModel):
    """End-to-end question answering response."""

    question: str
    answer: str
    sources: List[RAGSource]


class RAGService:
    """Coordinates document retrieval and LLM answer generation."""

    def __init__(
        self,
        retrieval_service: Optional[Any] = None,
        generation_service: Optional[GenerationService] = None,
    ) -> None:
        """Initialize RAGService with optional dependencies."""

        self.retrieval_service = retrieval_service or RetrievalService()
        self.generation_service = generation_service or GenerationService()

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGAnswer:
        """Answer a question using retrieved grounding context."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        cleaned_question = question.strip()

        retrieval_result = self.retrieval_service.retrieve(
            query=cleaned_question,
            top_k=top_k,
        )

        chunks = self._normalize_results(retrieval_result)

        prompt_contexts: List[PromptContext] = []
        sources: List[RAGSource] = []

        for chunk in chunks:
            metadata = (
                chunk.metadata
                if hasattr(chunk, "metadata") and chunk.metadata
                else {}
            )

            source_filename = str(
                metadata.get("source_filename", "") or ""
            )

            chunk_index_raw = metadata.get("chunk_index", -1)

            try:
                chunk_index = (
                    int(chunk_index_raw)
                    if chunk_index_raw is not None
                    else -1
                )
            except (ValueError, TypeError):
                chunk_index = -1

            prompt_contexts.append(
                PromptContext(
                    text=chunk.text,
                    source_filename=source_filename,
                    chunk_index=chunk_index,
                )
            )

            sources.append(
                RAGSource(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_filename=source_filename,
                    chunk_index=chunk_index,
                    distance=self._result_score(chunk),
                )
            )

        generated_answer = self.generation_service.generate(
            question=cleaned_question,
            contexts=prompt_contexts,
        )

        return RAGAnswer(
            question=cleaned_question,
            answer=generated_answer,
            sources=sources,
        )

    @staticmethod
    def _normalize_results(retrieval_result: Any) -> List[Any]:
        """Normalize supported retrieval result containers."""

        if hasattr(retrieval_result, "results"):
            return list(retrieval_result.results)

        if isinstance(retrieval_result, list):
            return retrieval_result

        if retrieval_result is None:
            return []

        raise ValueError("Unsupported retrieval result format.")

    @staticmethod
    def _result_score(result: Any) -> float:
        """Extract a numeric retrieval score from a result."""

        if hasattr(result, "distance"):
            return float(result.distance)

        if hasattr(result, "score"):
            return float(result.score)

        raise ValueError(
            "Retrieval result must provide either distance or score."
        )
