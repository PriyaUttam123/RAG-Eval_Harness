"""Tests for the document chunking service."""

import pytest

from backend.app.chunker import (
    Chunk,
    DocumentChunker,
    ChunkingError,
    InvalidChunkingParametersError,
    EmptyTextError,
)
from backend.app.document_loader import DocumentLoader, DocumentLoadResult, FileType


@pytest.fixture
def chunker_default() -> DocumentChunker:
    """Provide a DocumentChunker with default parameters."""
    return DocumentChunker(chunk_size=512, chunk_overlap=50)


@pytest.fixture
def chunker_small() -> DocumentChunker:
    """Provide a DocumentChunker with small chunk size for testing."""
    return DocumentChunker(chunk_size=20, chunk_overlap=5)


@pytest.fixture
def sample_document() -> DocumentLoadResult:
    """Provide a sample DocumentLoadResult for testing."""
    return DocumentLoadResult(
        text="This is a sample document for testing the chunking functionality. "
        "It contains multiple sentences to verify that the chunker works correctly.",
        source_filename="sample.txt",
        file_type=FileType.TXT,
        page_count=None,
        file_size_bytes=150,
    )


class TestChunkingParameters:
    """Tests for chunking parameter validation."""

    def test_default_parameters_are_valid(self) -> None:
        """Verify that default parameters create a valid chunker."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50

    def test_chunk_size_must_be_positive(self) -> None:
        """Verify that chunk_size <= 0 raises InvalidChunkingParametersError."""
        with pytest.raises(InvalidChunkingParametersError, match="chunk_size must be > 0"):
            DocumentChunker(chunk_size=0, chunk_overlap=0)

        with pytest.raises(InvalidChunkingParametersError, match="chunk_size must be > 0"):
            DocumentChunker(chunk_size=-10, chunk_overlap=0)

    def test_chunk_overlap_must_be_non_negative(self) -> None:
        """Verify that chunk_overlap < 0 raises InvalidChunkingParametersError."""
        with pytest.raises(InvalidChunkingParametersError, match="chunk_overlap must be >= 0"):
            DocumentChunker(chunk_size=100, chunk_overlap=-5)

    def test_chunk_overlap_must_be_less_than_chunk_size(self) -> None:
        """Verify that chunk_overlap >= chunk_size raises InvalidChunkingParametersError."""
        with pytest.raises(InvalidChunkingParametersError, match="chunk_overlap .* must be < chunk_size"):
            DocumentChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(InvalidChunkingParametersError, match="chunk_overlap .* must be < chunk_size"):
            DocumentChunker(chunk_size=50, chunk_overlap=100)

    def test_zero_overlap_is_allowed(self) -> None:
        """Verify that chunk_overlap = 0 is valid."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=0)
        assert chunker.chunk_overlap == 0

    def test_non_integer_parameters_raise_error(self) -> None:
        """Verify that non-integer types raise InvalidChunkingParametersError."""
        with pytest.raises(InvalidChunkingParametersError, match="must be an integer"):
            DocumentChunker(chunk_size=50.5, chunk_overlap=10)  # type: ignore

        with pytest.raises(InvalidChunkingParametersError, match="must be an integer"):
            DocumentChunker(chunk_size=50, chunk_overlap="10")  # type: ignore

        with pytest.raises(InvalidChunkingParametersError, match="must be an integer"):
            DocumentChunker(chunk_size=True, chunk_overlap=10)  # type: ignore


class TestShortTextChunking:
    """Tests for chunking short text that fits in one chunk."""

    def test_short_text_produces_one_chunk(self, chunker_default: DocumentChunker, sample_document: DocumentLoadResult) -> None:
        """Verify that text shorter than chunk_size produces exactly one chunk."""
        chunks = chunker_default.chunk(sample_document)
        assert len(chunks) == 1
        assert chunks[0].text == sample_document.text.strip()

    def test_single_chunk_has_correct_metadata(self, chunker_default: DocumentChunker, sample_document: DocumentLoadResult) -> None:
        """Verify that a single chunk has correct index, filename, and offsets."""
        chunks = chunker_default.chunk(sample_document)
        assert len(chunks) == 1
        chunk = chunks[0]

        assert chunk.chunk_index == 0
        assert chunk.source_filename == "sample.txt"
        assert chunk.start_offset == 0
        assert chunk.end_offset == len(sample_document.text)


class TestLongTextChunking:
    """Tests for chunking long text that requires multiple chunks."""

    def test_long_text_produces_multiple_chunks(self, chunker_small: DocumentChunker) -> None:
        """Verify that text longer than chunk_size produces multiple chunks."""
        long_text = "This is a long text that should be split into multiple chunks. " * 10
        document = DocumentLoadResult(
            text=long_text,
            source_filename="long.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=500,
        )

        chunks = chunker_small.chunk(document)
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self, chunker_small: DocumentChunker) -> None:
        """Verify that chunk indices are sequential starting from 0."""
        long_text = "This is a long text that should be split into multiple chunks. " * 10
        document = DocumentLoadResult(
            text=long_text,
            source_filename="long.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=500,
        )

        chunks = chunker_small.chunk(document)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_source_filename_is_preserved_across_chunks(self, chunker_small: DocumentChunker) -> None:
        """Verify that source filename is preserved in all chunks."""
        long_text = "This is a long text that should be split into multiple chunks. " * 10
        document = DocumentLoadResult(
            text=long_text,
            source_filename="document.pdf",
            file_type=FileType.PDF,
            page_count=5,
            file_size_bytes=500,
        )

        chunks = chunker_small.chunk(document)
        for chunk in chunks:
            assert chunk.source_filename == "document.pdf"


class TestOverlapCalculation:
    """Tests for chunk overlap functionality."""

    def test_overlap_is_correct(self) -> None:
        """Verify that overlap between consecutive chunks is correct."""
        chunker = DocumentChunker(chunk_size=30, chunk_overlap=10)
        text = "A" * 50  # 50 characters
        document = DocumentLoadResult(
            text=text,
            source_filename="overlap.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=50,
        )

        chunks = chunker.chunk(document)
        # With stride = 20 (30 - 10), text of 50 chars produces 3 chunks:
        # Chunk 0: 0-30, Chunk 1: 20-50, Chunk 2: 40-50 (partial)
        assert len(chunks) == 3

        # First chunk should be characters 0-30
        assert chunks[0].text == "A" * 30
        # Second chunk should start at position 20 (30 - 10 overlap)
        assert chunks[1].text == "A" * 30
        # Third chunk should be the remaining characters
        assert chunks[2].text == "A" * 10

    def test_no_overlap_when_overlap_is_zero(self) -> None:
        """Verify that chunks have no overlap when chunk_overlap = 0."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=0)
        text = "A" * 50
        document = DocumentLoadResult(
            text=text,
            source_filename="no_overlap.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=50,
        )

        chunks = chunker.chunk(document)
        assert len(chunks) == 3

        # Verify no overlap
        assert chunks[0].end_offset == chunks[1].start_offset
        assert chunks[1].end_offset == chunks[2].start_offset


class TestCharacterOffsets:
    """Tests for character offset accuracy."""

    def test_offsets_identify_chunk_in_original_text(self, chunker_small: DocumentChunker) -> None:
        """Verify that character offsets accurately identify each chunk's text."""
        text = "The quick brown fox jumps over the lazy dog."
        document = DocumentLoadResult(
            text=text,
            source_filename="offsets.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=44,
        )

        chunks = chunker_small.chunk(document)

        for chunk in chunks:
            # Extract text using offsets
            extracted = text[chunk.start_offset:chunk.end_offset]
            # The chunk text should match the extracted text
            assert extracted == chunk.text

    def test_offsets_are_monotonic(self, chunker_small: DocumentChunker) -> None:
        """Verify that offsets are monotonically increasing."""
        long_text = "This is a long text that should be split into multiple chunks. " * 10
        document = DocumentLoadResult(
            text=long_text,
            source_filename="monotonic.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=500,
        )

        chunks = chunker_small.chunk(document)

        for i in range(len(chunks) - 1):
            assert chunks[i].start_offset < chunks[i + 1].start_offset
            assert chunks[i].end_offset <= chunks[i + 1].end_offset


class TestEmptyAndWhitespaceHandling:
    """Tests for handling empty and whitespace-only text."""

    def test_empty_text_raises_error(self, chunker_default: DocumentChunker) -> None:
        """Verify that empty text raises EmptyTextError."""
        document = DocumentLoadResult(
            text="",
            source_filename="empty.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=0,
        )

        with pytest.raises(EmptyTextError, match="contains no text to chunk"):
            chunker_default.chunk(document)

    def test_whitespace_only_text_raises_error(self, chunker_default: DocumentChunker) -> None:
        """Verify that whitespace-only text raises EmptyTextError."""
        document = DocumentLoadResult(
            text="   \n\n  \t  ",
            source_filename="whitespace.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=10,
        )

        with pytest.raises(EmptyTextError, match="contains no text to chunk"):
            chunker_default.chunk(document)

    def test_text_with_leading_trailing_whitespace_is_chunked(self, chunker_default: DocumentChunker) -> None:
        """Verify that text with leading/trailing whitespace is chunked correctly."""
        document = DocumentLoadResult(
            text="   This is text with surrounding whitespace.   ",
            source_filename="whitespace_trimmed.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=50,
        )

        chunks = chunker_default.chunk(document)
        assert len(chunks) == 1
        # Text should be stripped
        assert chunks[0].text == "This is text with surrounding whitespace."


class TestDeterminism:
    """Tests for chunking determinism."""

    def test_chunking_same_input_twice_proces_same_output(self, chunker_default: DocumentChunker, sample_document: DocumentLoadResult) -> None:
        """Verify that chunking the same input twice produces identical output."""
        chunks1 = chunker_default.chunk(sample_document)
        chunks2 = chunker_default.chunk(sample_document)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_index == c2.chunk_index
            assert c1.source_filename == c2.source_filename
            assert c1.text == c2.text
            assert c1.start_offset == c2.start_offset
            assert c1.end_offset == c2.end_offset


class TestChunkModelValidation:
    """Tests for Chunk model validation."""

    def test_chunk_index_must_be_non_negative(self) -> None:
        """Verify that chunk_index < 0 raises validation error."""
        with pytest.raises(ValueError, match="chunk_index"):
            Chunk(
                chunk_index=-1,
                source_filename="test.txt",
                text="test",
                start_offset=0,
                end_offset=4,
            )

    def test_end_offset_must_be_greater_than_start_offset(self) -> None:
        """Verify that end_offset <= start_offset raises validation error."""
        with pytest.raises(ValueError, match="end_offset must be greater than start_offset"):
            Chunk(
                chunk_index=0,
                source_filename="test.txt",
                text="test",
                start_offset=10,
                end_offset=5,
            )

    def test_offsets_must_be_non_negative(self) -> None:
        """Verify that negative offsets raise validation error."""
        with pytest.raises(ValueError, match="start_offset"):
            Chunk(
                chunk_index=0,
                source_filename="test.txt",
                text="test",
                start_offset=-1,
                end_offset=4,
            )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_text_exactly_chunk_size_produces_one_chunk(self) -> None:
        """Verify that text exactly equal to chunk_size produces one chunk."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        text = "A" * 20
        document = DocumentLoadResult(
            text=text,
            source_filename="exact.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=20,
        )

        chunks = chunker.chunk(document)
        assert len(chunks) == 1

    def test_text_just_over_chunk_size_produces_two_chunks(self) -> None:
        """Verify that text just over chunk_size produces two chunks."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        text = "A" * 21
        document = DocumentLoadResult(
            text=text,
            source_filename="just_over.txt",
            file_type=FileType.TXT,
            page_count=None,
            file_size_bytes=21,
        )

        chunks = chunker.chunk(document)
        assert len(chunks) == 2


class TestDocumentIntegrationAndProvenance:
    """Tests for document loader integration and provenance tracking."""

    def test_invalid_document_type_raises_error(self, chunker_default: DocumentChunker) -> None:
        """Verify that passing an invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Expected DocumentLoadResult"):
            chunker_default.chunk("Just a string"  # type: ignore
            )

    def test_chunk_metadata_provenance(self, chunker_default: DocumentChunker, sample_document: DocumentLoadResult) -> None:
        """Verify that chunk metadata preserves source document provenance."""
        chunks = chunker_default.chunk(sample_document)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["source_filename"] == sample_document.source_filename
            assert chunk.metadata["file_type"] == sample_document.file_type.value
            assert chunk.metadata["page_count"] == sample_document.page_count
            assert chunk.metadata["file_size_bytes"] == sample_document.file_size_bytes

    def test_integration_with_document_loader(self, tmp_path) -> None:
        """Verify that DocumentChunker accepts DocumentLoadResult from DocumentLoader."""
        # Create a real text file
        test_file = tmp_path / "integration_doc.txt"
        content = "Line one: Introduction to RAG Evaluation Harness.\nLine two: Document chunking is tested."
        test_file.write_text(content, encoding="utf-8")

        # Load with DocumentLoader
        loader = DocumentLoader()
        load_result = loader.load(test_file)

        # Chunk with DocumentChunker
        chunker = DocumentChunker(chunk_size=40, chunk_overlap=10)
        chunks = chunker.chunk(load_result)

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.source_filename == "integration_doc.txt"
            # Verify slice matches
            assert load_result.text[chunk.start_offset:chunk.end_offset] == chunk.text
            assert chunk.metadata["source_filename"] == "integration_doc.txt"
            assert chunk.metadata["file_type"] == "txt"
