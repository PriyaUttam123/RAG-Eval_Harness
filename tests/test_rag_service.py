"""Unit tests for RAGService orchestration layer."""

from typing import Any, List, Optional

import pytest

from backend.app.prompt_builder import PromptContext
from backend.app.rag_service import RAGAnswer, RAGService, RAGSource
from backend.app.retrieval_service import RetrievalResult, RetrievalService, RetrievedChunk
from backend.app.generation_service import GenerationService


class FakeRetrievalService:
    """Fake retrieval service for testing orchestration without Chroma or embeddings."""

    def __init__(
        self,
        result: Optional[RetrievalResult] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self.result = result
        self.error = error
        self.last_query: Optional[str] = None
        self.last_top_k: Optional[int] = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        self.last_query = query
        self.last_top_k = top_k
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
        return RetrievalResult(query=query, results=[])


class FakeGenerationService:
    """Fake generation service for testing orchestration without LLM or OpenAI."""

    def __init__(
        self,
        answer: str = "Fake generated answer.",
        error: Optional[Exception] = None,
    ) -> None:
        self.answer = answer
        self.error = error
        self.last_question: Optional[str] = None
        self.last_contexts: Optional[List[PromptContext]] = None

    def generate(self, question: str, contexts: List[PromptContext]) -> str:
        self.last_question = question
        self.last_contexts = contexts
        if self.error:
            raise self.error
        return self.answer


def _create_sample_chunks() -> List[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            text="First relevant excerpt.",
            metadata={
                "source_filename": "guide.pdf",
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": 22,
            },
            distance=0.15,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            text="Second relevant excerpt.",
            metadata={
                "source_filename": "manual.txt",
                "chunk_index": 3,
                "start_offset": 50,
                "end_offset": 74,
            },
            distance=0.32,
        ),
    ]


def test_successful_end_to_end_orchestration():
    """Verify retrieval results are transformed into prompt context and answer is returned."""
    chunks = _create_sample_chunks()
    retrieval = FakeRetrievalService(result=RetrievalResult(query="test query", results=chunks))
    generation = FakeGenerationService(answer="Synthesized grounded answer.")

    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    result = service.answer("What is in the guide?", top_k=2)

    assert isinstance(result, RAGAnswer)
    assert result.answer == "Synthesized grounded answer."
    assert result.question == "What is in the guide?"
    assert len(result.sources) == 2

    # Generation service must receive 2 prompt contexts
    assert generation.last_contexts is not None
    assert len(generation.last_contexts) == 2
    assert generation.last_contexts[0].text == "First relevant excerpt."
    assert generation.last_contexts[1].text == "Second relevant excerpt."


def test_question_whitespace_is_stripped():
    """Verify question whitespace is cleaned across retrieval, generation, and output."""
    retrieval = FakeRetrievalService()
    generation = FakeGenerationService()
    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    result = service.answer("   \n\t What are the refund terms?   \t ")

    assert result.question == "What are the refund terms?"
    assert retrieval.last_query == "What are the refund terms?"
    assert generation.last_question == "What are the refund terms?"


def test_top_k_forwarded_unchanged():
    """Verify top_k parameter is forwarded directly to RetrievalService."""
    retrieval = FakeRetrievalService()
    generation = FakeGenerationService()
    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    service.answer("Question 1", top_k=5)
    assert retrieval.last_top_k == 5

    service.answer("Question 2", top_k=None)
    assert retrieval.last_top_k is None


def test_retrieved_metadata_converted_correctly_into_prompt_context():
    """Verify PromptContext receives text, source_filename, and chunk_index from chunk metadata."""
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-abc",
            text="Contextual text content.",
            metadata={
                "source_filename": "terms_and_conditions.pdf",
                "chunk_index": 4,
            },
            distance=0.08,
        )
    ]
    retrieval = FakeRetrievalService(result=RetrievalResult(query="terms", results=chunks))
    generation = FakeGenerationService()
    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    service.answer("What are the terms?")

    contexts = generation.last_contexts
    assert contexts is not None
    assert len(contexts) == 1
    assert contexts[0].text == "Contextual text content."
    assert contexts[0].source_filename == "terms_and_conditions.pdf"
    assert contexts[0].chunk_index == 4


def test_sources_preserve_retrieval_order():
    """Verify returned RAGAnswer sources strictly match retrieval ordering."""
    chunks = [
        RetrievedChunk(chunk_id="id-1", text="One", metadata={"chunk_index": 1}, distance=0.1),
        RetrievedChunk(chunk_id="id-2", text="Two", metadata={"chunk_index": 2}, distance=0.2),
        RetrievedChunk(chunk_id="id-3", text="Three", metadata={"chunk_index": 3}, distance=0.3),
    ]
    retrieval = FakeRetrievalService(result=RetrievalResult(query="order", results=chunks))
    generation = FakeGenerationService()
    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    result = service.answer("Check order?")

    assert [s.chunk_id for s in result.sources] == ["id-1", "id-2", "id-3"]
    assert [s.text for s in result.sources] == ["One", "Two", "Three"]
    assert [s.distance for s in result.sources] == [0.1, 0.2, 0.3]


def test_zero_retrieved_chunks():
    """Verify GenerationService is called with empty context list when zero chunks retrieved."""
    retrieval = FakeRetrievalService(result=RetrievalResult(query="empty", results=[]))
    generation = FakeGenerationService(answer="I don't have enough information in the provided context.")
    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    result = service.answer("Unknown topic question?")

    assert generation.last_contexts == []
    assert result.sources == []
    assert result.answer == "I don't have enough information in the provided context."


def test_empty_question_raises_value_error():
    """Verify empty question raises ValueError."""
    service = RAGService(
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        service.answer("")


def test_whitespace_only_question_raises_value_error():
    """Verify whitespace-only question raises ValueError."""
    service = RAGService(
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        service.answer("   \n\t  ")


@pytest.mark.parametrize("invalid_question", [None, 123, ["question"], {"q": "text"}])
def test_non_string_question_raises_value_error(invalid_question):
    """Verify non-string questions raise ValueError."""
    service = RAGService(
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        service.answer(invalid_question)


def test_retrieval_errors_propagate():
    """Verify errors raised by RetrievalService propagate without being swallowed."""
    retrieval = FakeRetrievalService(error=RuntimeError("Vector database connection lost."))
    generation = FakeGenerationService()
    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    with pytest.raises(RuntimeError, match="Vector database connection lost"):
        service.answer("Question causing retrieval failure?")


def test_generation_errors_propagate():
    """Verify errors raised by GenerationService propagate without being swallowed."""
    chunks = _create_sample_chunks()
    retrieval = FakeRetrievalService(result=RetrievalResult(query="q", results=chunks))
    generation = FakeGenerationService(error=RuntimeError("NVIDIA generation model timeout."))
    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    with pytest.raises(RuntimeError, match="NVIDIA generation model timeout"):
        service.answer("Question causing generation failure?")


def test_rag_answer_contains_cleaned_question():
    """Verify returned RAGAnswer preserves the stripped question."""
    retrieval = FakeRetrievalService()
    generation = FakeGenerationService()
    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    result = service.answer("   What is machine learning?   ")

    assert result.question == "What is machine learning?"


def test_returned_source_records_attributes():
    """Verify returned source records contain chunk_id, text, source_filename, chunk_index, distance."""
    chunk = RetrievedChunk(
        chunk_id="chunk-42",
        text="Sample passage text.",
        metadata={
            "source_filename": "handbook.pdf",
            "chunk_index": 7,
            "start_offset": 100,
            "end_offset": 120,
        },
        distance=0.1234,
    )
    retrieval = FakeRetrievalService(result=RetrievalResult(query="test", results=[chunk]))
    generation = FakeGenerationService()
    service = RAGService(retrieval_service=retrieval, generation_service=generation)

    result = service.answer("Test question?")

    assert len(result.sources) == 1
    source: RAGSource = result.sources[0]
    assert source.chunk_id == "chunk-42"
    assert source.text == "Sample passage text."
    assert source.source_filename == "handbook.pdf"
    assert source.chunk_index == 7
    assert source.distance == pytest.approx(0.1234)


def test_default_services_instantiation():
    """Verify RAGService instantiates real default services lazily without side effects."""
    service = RAGService()

    assert isinstance(service.retrieval_service, RetrievalService)
    assert isinstance(service.generation_service, GenerationService)
