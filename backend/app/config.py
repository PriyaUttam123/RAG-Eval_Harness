"""Application configuration module using Pydantic Settings.

Reads configuration from environment variables or a local `.env` file with
safe fallback defaults for local development.
"""

from functools import lru_cache
from typing import List, Union
from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the FastAPI backend service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = Field(
        default="RAG Evaluation Harness",
        description="Application display name",
    )
    version: str = Field(
        default="0.1.0",
        description="Backend API version",
    )
    environment: str = Field(
        default="development",
        description="Runtime environment: development, staging, or production",
    )
    log_level: str = Field(
        default="INFO",
        description="Application logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    backend_host: str = Field(
        default="0.0.0.0",
        description="Host interface for server binding",
    )
    backend_port: int = Field(
        default=8000,
        description="Port number for server binding",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Allowed CORS origins list",
    )
    max_file_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum allowed document file size in bytes (default: 10 MB)",
    )
    default_chunk_size: int = Field(
        default=512,
        description="Default chunk size in characters for document chunking",
        gt=0,
    )
    default_chunk_overlap: int = Field(
        default=50,
        description="Default chunk overlap in characters for document chunking",
        ge=0,
    )

    @field_validator("default_chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, overlap: int, info: ValidationInfo) -> int:
        """Validate that default_chunk_overlap is strictly less than default_chunk_size."""
        size = info.data.get("default_chunk_size", 512)
        if overlap >= size:
            raise ValueError(
                f"default_chunk_overlap ({overlap}) must be less than default_chunk_size ({size})."
            )
        return overlap

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Union[str, list[str]]) -> list[str]:
        """Support comma-separated strings or JSON arrays in environment variables."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                import json
                return json.loads(value)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_for_production(cls, origins: list[str], info: ValidationInfo) -> list[str]:
        """Prevent unrestricted wildcard origins in production environments."""
        env = info.data.get("environment", "development")
        if env == "production" and "*" in origins:
            raise ValueError("Wildcard '*' CORS origin is not permitted in production environment.")
        return origins


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the Settings."""
    return Settings()
