"""Document loading service for extracting text and metadata from files.

This module provides a reusable service for loading and extracting text content
from .txt and .pdf files. It is designed as an internal Python service (not a
public HTTP endpoint) to be used by other backend components.

Supported formats:
- .txt: Plain text files with explicit encoding handling
- .pdf: PDF documents with page-by-page text extraction

The loader validates file size, handles errors gracefully, and returns
structured, typed results with metadata.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

try:
    from backend.app.config import get_settings
except ImportError:
    from app.config import get_settings


class FileType(str, Enum):
    """Supported file types for document loading."""

    TXT = "txt"
    PDF = "pdf"


class DocumentLoadResult(BaseModel):
    """Structured result from document loading operation.

    Attributes:
        text: Extracted text content from the document
        source_filename: Original name of the source file
        file_type: Type of file that was loaded (txt or pdf)
        page_count: Number of pages (None for non-page-based formats like txt)
        file_size_bytes: Size of the file in bytes
    """

    text: str = Field(..., description="Extracted text content from the document")
    source_filename: str = Field(..., description="Original name of the source file")
    file_type: FileType = Field(..., description="Type of file that was loaded")
    page_count: Optional[int] = Field(
        None, description="Number of pages (None for non-page-based formats)"
    )
    file_size_bytes: int = Field(..., description="Size of the file in bytes")


class DocumentLoadError(Exception):
    """Base exception for document loading errors."""

    pass


class FileNotFoundError(DocumentLoadError):
    """Raised when the specified file does not exist."""

    pass


class UnsupportedFileTypeError(DocumentLoadError):
    """Raised when the file extension is not supported."""

    pass


class FileSizeExceededError(DocumentLoadError):
    """Raised when the file exceeds the maximum allowed size."""

    pass


class EmptyFileError(DocumentLoadError):
    """Raised when the file is empty or contains no extractable text."""

    pass


class CorruptFileError(DocumentLoadError):
    """Raised when the file is corrupted or unreadable."""

    pass


class DocumentLoader:
    """Service for loading and extracting text from document files.

    This class provides methods to safely load .txt and .pdf files,
    validate their size and integrity, and extract text content with
    associated metadata.

    Usage:
        >>> loader = DocumentLoader()
        >>> result = loader.load("path/to/document.pdf")
        >>> print(result.text)
        >>> print(result.page_count)
    """

    def __init__(self, max_file_size_bytes: Optional[int] = None):
        """Initialize the document loader.

        Args:
            max_file_size_bytes: Maximum allowed file size in bytes.
                If None, uses the value from application settings.
        """
        if max_file_size_bytes is None:
            settings = get_settings()
            self.max_file_size_bytes = settings.max_file_size_bytes
        else:
            self.max_file_size_bytes = max_file_size_bytes

    def load(self, file_path: str | Path) -> DocumentLoadResult:
        """Load a document file and extract its text and metadata.

        Args:
            file_path: Path to the document file (string or Path object)

        Returns:
            DocumentLoadResult containing extracted text and metadata

        Raises:
            FileNotFoundError: If the file does not exist
            UnsupportedFileTypeError: If the file extension is not supported
            FileSizeExceededError: If the file exceeds the maximum size
            EmptyFileError: If the file is empty or has no extractable text
            CorruptFileError: If the file is corrupted or unreadable
        """
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {file_path}")

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise FileSizeExceededError(
                f"File size ({file_size} bytes) exceeds maximum allowed "
                f"({self.max_file_size_bytes} bytes)"
            )

        # Determine file type (case-insensitive)
        extension = path.suffix.lower()

        if extension == ".txt":
            return self._load_txt(path, file_size)
        elif extension == ".pdf":
            return self._load_pdf(path, file_size)
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported file extension: {extension}. "
                f"Supported extensions: .txt, .pdf"
            )

    def _load_txt(self, path: Path, file_size: int) -> DocumentLoadResult:
        """Load and extract text from a .txt file.

        Args:
            path: Path to the text file
            file_size: Size of the file in bytes

        Returns:
            DocumentLoadResult with extracted text and metadata

        Raises:
            EmptyFileError: If the file is empty
            CorruptFileError: If the file cannot be read
        """
        try:
            # Try UTF-8 first (most common), then fallback to latin-1
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")

            # Check for empty content
            if not text or not text.strip():
                raise EmptyFileError(f"File is empty or contains no text: {path}")

            return DocumentLoadResult(
                text=text.strip(),
                source_filename=path.name,
                file_type=FileType.TXT,
                page_count=None,
                file_size_bytes=file_size,
            )

        except (OSError, IOError) as e:
            raise CorruptFileError(f"Failed to read text file: {e}")

    def _load_pdf(self, path: Path, file_size: int) -> DocumentLoadResult:
        """Load and extract text from a .pdf file.

        Args:
            path: Path to the PDF file
            file_size: Size of the file in bytes

        Returns:
            DocumentLoadResult with extracted text and metadata

        Raises:
            EmptyFileError: If the PDF has no extractable text
            CorruptFileError: If the PDF is corrupted or unreadable
        """
        try:
            import pypdf
        except ImportError:
            raise CorruptFileError(
                "pypdf library is not installed. "
                "Install it with: pip install pypdf"
            )

        try:
            reader = pypdf.PdfReader(path)
            page_count = len(reader.pages)

            if page_count == 0:
                raise EmptyFileError(f"PDF has no pages: {path}")

            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception:
                    # Skip pages that fail to extract, but continue with others
                    continue

            full_text = "\n\n".join(text_parts)

            # Check for empty content - only raise error if file is completely empty (0 bytes)
            # PDFs with blank pages are valid and should load with empty text
            if file_size == 0:
                raise EmptyFileError(f"PDF file is empty: {path}")

            return DocumentLoadResult(
                text=full_text.strip() if full_text else "",
                source_filename=path.name,
                file_type=FileType.PDF,
                page_count=page_count,
                file_size_bytes=file_size,
            )

        except pypdf.errors.EmptyFileError as e:
            # Convert pypdf's EmptyFileError to our EmptyFileError
            raise EmptyFileError(f"PDF file is empty or corrupted: {e}")
        except EmptyFileError:
            # Re-raise our own EmptyFileError as-is
            raise
        except Exception as e:
            # Handle various PDF-related errors
            error_msg = str(e).lower()
            if "pdf" in error_msg or "encrypted" in error_msg or "password" in error_msg:
                raise CorruptFileError(f"Corrupted or invalid PDF file: {e}")
            else:
                raise CorruptFileError(f"Failed to read PDF file: {e}")
