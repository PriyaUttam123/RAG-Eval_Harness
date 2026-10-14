"""Document chunking service for splitting text into smaller segments.

This module provides a reusable service for splitting document text into
smaller chunks with configurable size and overlap. It is designed as an
internal Python service (not a public HTTP endpoint) to be used by other
backend components during the document ingestion pipeline.

The chunker uses deterministic character-based splitting with:
- Configurable chunk size and overlap (with application settings fallback)
- Stable chunk indices (0, 1, 2, ...)
- Character offsets using Python slicing convention (start-inclusive, end-exclusive)
- Source filename and document metadata tracking
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

try:
    from backend.app.config import get_settings
    from backend.app.document_loader import DocumentLoadResult
except ImportError:
    from app.config import get_settings
    from app.document_loader import DocumentLoadResult


class Chunk(BaseModel):
    """A single text chunk with metadata.

    Offsets follow the start-inclusive, end-exclusive Python slicing convention:
    `original_text[start_offset:end_offset] == text`.

    Attributes:
        chunk_index: Zero-based index of this chunk within its source document
        source_filename: Original name of the source file
        text: The chunk's text content
        start_offset: Start character offset in the original text (inclusive)
        end_offset: End character offset in the original text (exclusive)
        metadata: Tracing metadata dictionary containing source document provenance
    """

    chunk_index: int = Field(
        ...,
        description="Zero-based index of this chunk within its source document",
        ge=0,
    )
    source_filename: str = Field(
        ...,
        description="Original name of the source file",
    )
    text: str = Field(
        ...,
        description="The chunk's text content",
    )
    start_offset: int = Field(
        ...,
        description="Start character offset in original text (inclusive)",
        ge=0,
    )
    end_offset: int = Field(
        ...,
        description="End character offset in original text (exclusive)",
        ge=0,
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance and source metadata dictionary",
    )

    @field_validator("end_offset")
    @classmethod
    def end_offset_greater_than_start(cls, v: int, info) -> int:
        """Ensure end_offset is greater than start_offset."""
        if "start_offset" in info.data and v <= info.data["start_offset"]:
            raise ValueError("end_offset must be greater than start_offset")
        return v


class ChunkingError(Exception):
    """Base exception for chunking errors."""

    pass


class InvalidChunkingParametersError(ChunkingError):
    """Raised when chunk_size or chunk_overlap parameters are invalid."""

    pass


class EmptyTextError(ChunkingError):
    """Raised when the input text is empty or contains only whitespace."""

    pass


class DocumentChunker:
    """Service for splitting document text into chunks.

    This class provides deterministic character-based text chunking with
    configurable size and overlap. It accepts DocumentLoadResult from the
    document loader and returns a list of Chunk objects with metadata.

    Usage:
        >>> chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
        >>> chunks = chunker.chunk(document_load_result)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.chunk_index}: {chunk.text[:50]}...")
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Initialize the document chunker.

        Args:
            chunk_size: Maximum number of characters per chunk (must be > 0).
                If None, uses default_chunk_size from application settings (512).
            chunk_overlap: Number of characters to overlap between chunks
                (must be >= 0 and < chunk_size).
                If None, uses default_chunk_overlap from application settings (50).

        Raises:
            InvalidChunkingParametersError: If parameters are invalid
        """
        settings = get_settings()
        actual_size = settings.default_chunk_size if chunk_size is None else chunk_size
        actual_overlap = settings.default_chunk_overlap if chunk_overlap is None else chunk_overlap

        self._validate_parameters(actual_size, actual_overlap)
        self.chunk_size = actual_size
        self.chunk_overlap = actual_overlap

    def _validate_parameters(self, chunk_size: Any, chunk_overlap: Any) -> None:
        """Validate chunking parameters.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks

        Raises:
            InvalidChunkingParametersError: If parameters are invalid
        """
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise InvalidChunkingParametersError(
                f"chunk_size must be an integer, got {type(chunk_size).__name__}"
            )
        if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
            raise InvalidChunkingParametersError(
                f"chunk_overlap must be an integer, got {type(chunk_overlap).__name__}"
            )
        if chunk_size <= 0:
            raise InvalidChunkingParametersError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise InvalidChunkingParametersError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise InvalidChunkingParametersError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )

    def chunk(self, document: DocumentLoadResult) -> List[Chunk]:
        """Split a document's text into chunks.

        Args:
            document: DocumentLoadResult from the document loader

        Returns:
            List of Chunk objects with text and metadata

        Raises:
            TypeError: If document is not an instance of DocumentLoadResult
            EmptyTextError: If the document text is empty or whitespace-only
        """
        if not isinstance(document, DocumentLoadResult):
            raise TypeError(
                f"Expected DocumentLoadResult, got {type(document).__name__}"
            )

        text = document.text

        # Check for empty or whitespace-only text
        if not text or not text.strip():
            raise EmptyTextError(
                f"Document '{document.source_filename}' contains no text to chunk"
            )

        # Source metadata to trace back to source document
        source_metadata: Dict[str, Any] = {
            "source_filename": document.source_filename,
            "file_type": document.file_type.value if hasattr(document.file_type, "value") else str(document.file_type),
            "page_count": document.page_count,
            "file_size_bytes": document.file_size_bytes,
        }

        # Determine leading whitespace offset in the original document text
        leading_ws = len(text) - len(text.lstrip())
        stripped_text = text.strip()

        # Calculate stride (advance distance between consecutive chunks)
        stride = self.chunk_size - self.chunk_overlap

        chunks: List[Chunk] = []
        chunk_index = 0

        # If text fits within a single chunk, return immediately
        if len(stripped_text) <= self.chunk_size:
            chunk = Chunk(
                chunk_index=0,
                source_filename=document.source_filename,
                text=stripped_text,
                start_offset=leading_ws,
                end_offset=leading_ws + len(stripped_text),
                metadata=source_metadata,
            )
            chunks.append(chunk)
            return chunks

        # Character-based sliding window with overlap for longer texts
        start = 0
        while start < len(stripped_text):
            end = min(start + self.chunk_size, len(stripped_text))

            if start == end:
                break

            chunk_text = stripped_text[start:end]

            # Avoid unnecessary empty/whitespace-only chunks
            if not chunk_text.strip():
                start += stride
                continue

            original_start = leading_ws + start
            original_end = leading_ws + end

            chunk = Chunk(
                chunk_index=chunk_index,
                source_filename=document.source_filename,
                text=chunk_text,
                start_offset=original_start,
                end_offset=original_end,
                metadata=source_metadata,
            )
            chunks.append(chunk)

            start += stride
            chunk_index += 1

            if start >= len(stripped_text):
                break

            # Defensive safety guard against infinite loops
            if stride <= 0:
                break

        return chunks
