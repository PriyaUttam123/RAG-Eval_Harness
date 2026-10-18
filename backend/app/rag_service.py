"""RAG orchestration service coordinating retrieval and generation."""

from typing import List, Optional

from pydantic import BaseModel

try:
    from backend.app.generation_service import GenerationService
    from backend.app.prompt_builder import PromptContext
    from backend.app.retrieval_service import RetrievalResult, RetrievalService, RetrievedChunk
except ImportError:
    from app.generation_service import GenerationService
    from app.prompt_builder import PromptContext
    from app.retrieval_service import RetrievalResult, RetrievalService, RetrievedChunk


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
        retrieval_service: Optional[RetrievalService] = None,
        generation_service: Optional[GenerationService] = None,
    ) -> None:
        """Initialize RAGService with optional dependencies.

        Args:
            retrieval_service: Optional RetrievalService instance for chunk search.
            generation_service: Optional GenerationService instance for response synthesis.
        """
        self.retrieval_service = retrieval_service or RetrievalService()
        self.generation_service = generation_service or GenerationService()

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGAnswer:
        """Answer a question using retrieved grounding context.

        Args:
            question: The user query string.
            top_k: Optional number of top similar chunks to retrieve.

        Returns:
            RAGAnswer containing cleaned question, generated answer, and sources.

        Raises:
            ValueError: If the question is not a non-empty string.
            RuntimeError: If retrieval or generation fails.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        cleaned_question = question.strip()

        retrieval_result = self.retrieval_service.retrieve(
            query=cleaned_question,
            top_k=top_k,
        )

        chunks: List[RetrievedChunk] = (
            retrieval_result.results
            if hasattr(retrieval_result, "results")
            else (retrieval_result or [])
        )

        prompt_contexts: List[PromptContext] = []
        sources: List[RAGSource] = []

        for chunk in chunks:
            metadata = chunk.metadata if hasattr(chunk, "metadata") and chunk.metadata else {}

            source_filename = str(metadata.get("source_filename", "") or "")

            chunk_index_raw = metadata.get("chunk_index", -1)
            try:
                chunk_index = int(chunk_index_raw) if chunk_index_raw is not None else -1
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
                    distance=float(chunk.distance),
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
