"""Tests for the document loading service."""

import pytest
from pathlib import Path

from backend.app.document_loader import (
    DocumentLoader,
    DocumentLoadResult,
    FileType,
    FileNotFoundError,
    UnsupportedFileTypeError,
    FileSizeExceededError,
    EmptyFileError,
    CorruptFileError,
)


@pytest.fixture
def loader() -> DocumentLoader:
    """Provide a DocumentLoader instance with default settings."""
    return DocumentLoader()


@pytest.fixture
def loader_with_small_limit() -> DocumentLoader:
    """Provide a DocumentLoader instance with a small file size limit for testing."""
    return DocumentLoader(max_file_size_bytes=100)


@pytest.fixture
def temp_txt_file(tmp_path: Path) -> Path:
    """Create a temporary text file with sample content."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a sample text file for testing.", encoding="utf-8")
    return file_path


@pytest.fixture
def temp_empty_txt_file(tmp_path: Path) -> Path:
    """Create a temporary empty text file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def temp_whitespace_txt_file(tmp_path: Path) -> Path:
    """Create a temporary text file with only whitespace."""
    file_path = tmp_path / "whitespace.txt"
    file_path.write_text("   \n\n  \t  ", encoding="utf-8")
    return file_path


@pytest.fixture
def temp_pdf_file(tmp_path: Path) -> Path:
    """Create a temporary PDF file with extractable text."""
    try:
        import pypdf

        file_path = tmp_path / "sample.pdf"
        writer = pypdf.PdfWriter()

        # Add a blank page (pypdf can't add text directly, but we can test the structure)
        page = pypdf.PageObject.create_blank_page(width=612, height=792)
        writer.add_page(page)

        # Write the PDF
        with open(file_path, "wb") as f:
            writer.write(f)

        return file_path
    except ImportError:
        pytest.skip("pypdf not installed")


@pytest.fixture
def temp_large_txt_file(tmp_path: Path) -> Path:
    """Create a temporary text file that exceeds the small size limit."""
    file_path = tmp_path / "large.txt"
    # Create content larger than 100 bytes
    file_path.write_text("x" * 200, encoding="utf-8")
    return file_path


class TestSuccessfulTXTExtraction:
    """Tests for successful TXT file extraction."""

    def test_load_txt_returns_correct_result(self, loader: DocumentLoader, temp_txt_file: Path) -> None:
        """Verify that loading a TXT file returns correct text and metadata."""
        result = loader.load(temp_txt_file)

        assert isinstance(result, DocumentLoadResult)
        assert result.text == "This is a sample text file for testing."
        assert result.source_filename == "sample.txt"
        assert result.file_type == FileType.TXT
        assert result.page_count is None
        assert result.file_size_bytes > 0

    def test_load_txt_case_insensitive_extension(self, loader: DocumentLoader, tmp_path: Path) -> None:
        """Verify that file extensions are handled case-insensitively."""
        # Create file with uppercase extension
        file_path = tmp_path / "SAMPLE.TXT"
        file_path.write_text("Test content", encoding="utf-8")

        result = loader.load(file_path)
        assert result.file_type == FileType.TXT
        assert result.text == "Test content"

    def test_load_txt_with_latin1_encoding(self, loader: DocumentLoader, tmp_path: Path) -> None:
        """Verify that TXT files with latin-1 encoding are handled."""
        file_path = tmp_path / "latin1.txt"
        # Write with latin-1 encoding
        file_path.write_text("Test content with special chars: éàü", encoding="latin-1")

        result = loader.load(file_path)
        assert result.text == "Test content with special chars: éàü"


class TestSuccessfulPDFExtraction:
    """Tests for successful PDF file extraction."""

    def test_load_pdf_returns_page_count(self, loader: DocumentLoader, temp_pdf_file: Path) -> None:
        """Verify that loading a PDF returns correct page count."""
        result = loader.load(temp_pdf_file)

        assert isinstance(result, DocumentLoadResult)
        assert result.source_filename == "sample.pdf"
        assert result.file_type == FileType.PDF
        assert result.page_count == 1
        assert result.file_size_bytes > 0
        # Note: Text will be empty since pypdf creates blank pages
        # In production, use real PDFs with actual text content

    def test_load_pdf_case_insensitive_extension(self, loader: DocumentLoader, tmp_path: Path) -> None:
        """Verify that PDF extensions are handled case-insensitively."""
        try:
            import pypdf

            file_path = tmp_path / "SAMPLE.PDF"
            writer = pypdf.PdfWriter()
            page = pypdf.PageObject.create_blank_page(width=612, height=792)
            writer.add_page(page)

            with open(file_path, "wb") as f:
                writer.write(f)

            result = loader.load(file_path)
            assert result.file_type == FileType.PDF
            assert result.page_count == 1
        except ImportError:
            pytest.skip("pypdf not installed")


class TestErrorHandling:
    """Tests for error handling in document loading."""

    def test_missing_file_raises_error(self, loader: DocumentLoader) -> None:
        """Verify that loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load("nonexistent_file.txt")

    def test_directory_path_raises_error(self, loader: DocumentLoader, tmp_path: Path) -> None:
        """Verify that loading a directory path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not a file"):
            loader.load(tmp_path)

    def test_unsupported_extension_raises_error(self, loader: DocumentLoader, tmp_path: Path) -> None:
        """Verify that unsupported file extensions raise UnsupportedFileTypeError."""
        file_path = tmp_path / "document.docx"
        file_path.write_text("fake content", encoding="utf-8")

        with pytest.raises(UnsupportedFileTypeError, match="Unsupported file extension"):
            loader.load(file_path)

    def test_empty_txt_file_raises_error(self, loader: DocumentLoader, temp_empty_txt_file: Path) -> None:
        """Verify that empty TXT files raise EmptyFileError."""
        with pytest.raises(EmptyFileError, match="empty or contains no text"):
            loader.load(temp_empty_txt_file)

    def test_whitespace_only_txt_raises_error(self, loader: DocumentLoader, temp_whitespace_txt_file: Path) -> None:
        """Verify that TXT files with only whitespace raise EmptyFileError."""
        with pytest.raises(EmptyFileError, match="empty or contains no text"):
            loader.load(temp_whitespace_txt_file)

    def test_oversized_file_raises_error(self, loader_with_small_limit: DocumentLoader, temp_large_txt_file: Path) -> None:
        """Verify that files exceeding the size limit raise FileSizeExceededError."""
        with pytest.raises(FileSizeExceededError, match="exceeds maximum allowed"):
            loader_with_small_limit.load(temp_large_txt_file)

    def test_corrupt_pdf_raises_error(self, loader: DocumentLoader, tmp_path: Path) -> None:
        """Verify that corrupt PDF files raise CorruptFileError."""
        file_path = tmp_path / "corrupt.pdf"
        # Write invalid PDF content
        file_path.write_bytes(b"This is not a valid PDF file")

        with pytest.raises(CorruptFileError):
            loader.load(file_path)


class TestPDFWithNoExtractableText:
    """Tests for PDFs with no extractable text."""

    def test_pdf_with_blank_pages_loads_with_empty_text(self, loader: DocumentLoader, tmp_path: Path) -> None:
        """Verify that PDFs with blank pages load successfully with empty text."""
        try:
            import pypdf

            file_path = tmp_path / "blank.pdf"
            writer = pypdf.PdfWriter()
            page = pypdf.PageObject.create_blank_page(width=612, height=792)
            writer.add_page(page)

            with open(file_path, "wb") as f:
                writer.write(f)

            # This PDF has a page but no extractable text - should load with empty text
            result = loader.load(file_path)
            assert result.text == ""
            assert result.page_count == 1
        except ImportError:
            pytest.skip("pypdf not installed")

    def test_truly_empty_pdf_raises_error(self, loader: DocumentLoader, tmp_path: Path) -> None:
        """Verify that a 0-byte PDF file raises EmptyFileError."""
        file_path = tmp_path / "empty.pdf"
        file_path.write_bytes(b"")

        with pytest.raises(EmptyFileError):
            loader.load(file_path)


class TestCustomMaxFileSize:
    """Tests for custom max file size configuration."""

    def test_custom_max_size_is_respected(self, tmp_path: Path) -> None:
        """Verify that custom max file size is used when provided."""
        # Create a file of 50 bytes
        file_path = tmp_path / "medium.txt"
        file_path.write_text("x" * 50, encoding="utf-8")

        # Loader with limit of 100 bytes should accept it
        loader = DocumentLoader(max_file_size_bytes=100)
        result = loader.load(file_path)
        assert result.text == "x" * 50

        # Loader with limit of 25 bytes should reject it
        loader_small = DocumentLoader(max_file_size_bytes=25)
        with pytest.raises(FileSizeExceededError):
            loader_small.load(file_path)


class TestFileMetadataAccuracy:
    """Tests for accurate file metadata in results."""

    def test_txt_file_size_is_accurate(self, loader: DocumentLoader, temp_txt_file: Path) -> None:
        """Verify that the reported file size matches the actual file size."""
        result = loader.load(temp_txt_file)
        actual_size = temp_txt_file.stat().st_size
        assert result.file_size_bytes == actual_size

    def test_source_filename_preserves_original_name(self, loader: DocumentLoader, tmp_path: Path) -> None:
        """Verify that the source filename preserves the original file name."""
        file_path = tmp_path / "my_document.txt"
        file_path.write_text("Content", encoding="utf-8")

        result = loader.load(file_path)
        assert result.source_filename == "my_document.txt"
