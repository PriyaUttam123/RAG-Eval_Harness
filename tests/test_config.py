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
