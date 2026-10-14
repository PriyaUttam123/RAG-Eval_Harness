"""Tests for application settings and configuration validation."""

import pytest
from backend.app.config import Settings


def test_default_settings_initialization() -> None:
    """Verify that default settings use safe local values."""
    settings = Settings()
    assert settings.project_name == "RAG Evaluation Harness"
    assert settings.environment == "development"
    assert settings.backend_host == "0.0.0.0"
    assert settings.backend_port == 8000
    assert "http://localhost:5173" in settings.cors_origins
    assert "http://127.0.0.1:5173" in settings.cors_origins


def test_cors_origins_from_comma_separated_string() -> None:
    """Verify that comma-separated origin strings are parsed into a list."""
    settings = Settings(cors_origins="http://example.com, https://app.example.com")
    assert settings.cors_origins == ["http://example.com", "https://app.example.com"]


def test_cors_origins_from_json_string() -> None:
    """Verify that JSON array string origins are parsed into a list."""
    settings = Settings(cors_origins='["http://localhost:3000"]')
    assert settings.cors_origins == ["http://localhost:3000"]


def test_cors_production_wildcard_rejected() -> None:
    """Verify that wildcard origin is rejected when environment is set to production."""
    with pytest.raises(ValueError, match="Wildcard"):
        Settings(environment="production", cors_origins="*")


def test_chunking_settings_defaults() -> None:
    """Verify default chunk size and overlap values."""
    settings = Settings()
    assert settings.default_chunk_size == 512
    assert settings.default_chunk_overlap == 50


def test_chunking_settings_overlap_validation() -> None:
    """Verify that default_chunk_overlap >= default_chunk_size is rejected."""
    with pytest.raises(ValueError, match="must be less than default_chunk_size"):
        Settings(default_chunk_size=100, default_chunk_overlap=100)

    with pytest.raises(ValueError, match="must be less than default_chunk_size"):
        Settings(default_chunk_size=100, default_chunk_overlap=150)

def test_embedding_and_retrieval_settings_defaults() -> None:
    """Verify default embedding and vector retrieval configuration."""
    settings = Settings()
    assert settings.embedding_model_name == "all-MiniLM-L6-v2"
    assert settings.embedding_dimension == 384
    assert settings.chroma_persist_dir == "./data/chroma"
    assert settings.chroma_collection_name == "rag_eval_chunks"
    assert settings.chroma_distance_metric == "cosine"
    assert settings.default_retrieval_k == 4


@pytest.mark.parametrize("metric", ["cosine", "l2", "ip"])
def test_valid_chroma_distance_metrics(metric: str) -> None:
    """Accept all supported Chroma distance metrics."""
    settings = Settings(chroma_distance_metric=metric)
    assert settings.chroma_distance_metric == metric


def test_chroma_distance_metric_normalized_to_lowercase() -> None:
    """Normalize uppercase metric names."""
    settings = Settings(chroma_distance_metric="COSINE")
    assert settings.chroma_distance_metric == "cosine"


def test_invalid_chroma_distance_metric_rejected() -> None:
    """Reject unsupported distance metrics."""
    with pytest.raises(ValueError, match="Invalid chroma_distance_metric"):
        Settings(chroma_distance_metric="euclidean")


def test_empty_embedding_model_name_rejected() -> None:
    """Reject an empty embedding model name."""
    with pytest.raises(ValueError):
        Settings(embedding_model_name="")


def test_empty_chroma_collection_name_rejected() -> None:
    """Reject an empty Chroma collection name."""
    with pytest.raises(ValueError):
        Settings(chroma_collection_name="")


@pytest.mark.parametrize("dimension", [0, -1])
def test_invalid_embedding_dimension_rejected(dimension: int) -> None:
    """Require a positive embedding dimension."""
    with pytest.raises(ValueError):
        Settings(embedding_dimension=dimension)


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_retrieval_k_rejected(k: int) -> None:
    """Require a positive number of retrieved chunks."""
    with pytest.raises(ValueError):
        Settings(default_retrieval_k=k)
