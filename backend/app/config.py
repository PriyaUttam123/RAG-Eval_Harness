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
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Default embedding model identifier for chunk vectorization",
        min_length=1,
    )
    embedding_dimension: int = Field(
        default=384,
        description="Expected dimensionality of embedding vectors",
        gt=0,
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="Filesystem directory for Chroma local vector persistence",
    )
    chroma_collection_name: str = Field(
        default="rag_eval_chunks",
        description="Chroma collection name for document chunks",
        min_length=1,
    )
    chroma_distance_metric: str = Field(
        default="cosine",
        description="Distance metric for vector search space ('cosine', 'l2', or 'ip')",
    )
    default_retrieval_k: int = Field(
        default=4,
        description="Default number of top similar chunks to retrieve",
        gt=0,
    )

    # Generation / NVIDIA API configuration
    nvidia_api_key: str = Field(
        default="",
        description="NVIDIA API key for hosted generation",
    )
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="OpenAI-compatible NVIDIA API base URL",
    )
    nvidia_model_name: str = Field(
        default="",
        description="NVIDIA-hosted generation model identifier",
    )
    generation_temperature: float = Field(
        default=0.0,
        description="Generation temperature",
        ge=0.0,
    )
    generation_max_tokens: int = Field(
        default=512,
        description="Maximum number of tokens generated per response",
        gt=0,
    )

    @field_validator("chroma_distance_metric")
    @classmethod
    def validate_distance_metric(cls, metric: str) -> str:
        """Ensure Chroma distance metric is a valid HNSW space."""
        valid_metrics = {"cosine", "l2", "ip"}
        if metric.lower() not in valid_metrics:
            raise ValueError(
                f"Invalid chroma_distance_metric '{metric}'. Must be one of {valid_metrics}"
            )
        return metric.lower()

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
    def validate_cors_for_production(
        cls,
        origins: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        """Prevent unrestricted wildcard origins in production environments."""
        env = info.data.get("environment", "development")
        if env == "production" and "*" in origins:
            raise ValueError(
                "Wildcard '*' CORS origin is not permitted in production environment."
            )
        return origins


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the Settings."""
    return Settings()