"""Experiment configuration and run metadata models for RAG benchmarking."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ExperimentConfig(BaseModel):
    """Reproducible configuration specifying parameters for a RAG experiment."""

    name: str = Field(..., description="Unique human-readable experiment name.")
    description: str = Field(default="", description="Optional description of the experiment.")
    chunk_size: int = Field(..., gt=0, description="Chunk character size for document splitting.")
    chunk_overlap: int = Field(..., ge=0, description="Overlapping characters between adjacent chunks.")
    retrieval_top_k: int = Field(..., gt=0, description="Number of top chunks to retrieve.")
    evaluation_k: int = Field(..., gt=0, description="Cutoff k for retrieval quality metrics.")
    retrieval_strategy: str = Field(
        default="vector",
        description="Retrieval strategy ('vector' or 'hybrid').",
    )
    reranking_enabled: bool = Field(
        default=False,
        description="Whether a cross-encoder reranker is applied after retrieval.",
    )
    embedding_model: str = Field(
        ...,
        description="Embedding model identifier used for vector encoding.",
    )
    generation_model: str = Field(
        default="",
        description="LLM generation model identifier (optional for retrieval-only runs).",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("name must be a non-empty string.")
        return value.strip()

    @field_validator("description", "generation_model")
    @classmethod
    def strip_optional_strings(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("embedding_model must be a non-empty string.")
        return value.strip()

    @field_validator("retrieval_strategy")
    @classmethod
    def validate_retrieval_strategy(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("retrieval_strategy must be a string.")
        normalized = value.strip().lower()
        if normalized not in {"vector", "hybrid"}:
            raise ValueError(
                f"Invalid retrieval_strategy '{value}'. Must be one of ['vector', 'hybrid']."
            )
        return normalized

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, overlap: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})."
            )
        return overlap


class ExperimentRunMetadata(BaseModel):
    """Metadata for an executed experiment run against a dataset."""

    experiment_id: str = Field(..., description="Unique identifier for the experiment run.")
    config: ExperimentConfig = Field(..., description="The experiment configuration.")
    dataset_name: str = Field(..., description="Dataset evaluated ('golden' or 'holdout').")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timezone-aware timestamp of when the run metadata was created.",
    )

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("experiment_id must be a non-empty string.")
        return value.strip()

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("dataset_name must be a string.")
        normalized = value.strip().lower()
        if normalized not in {"golden", "holdout"}:
            raise ValueError(
                f"Invalid dataset_name '{value}'. Must be 'golden' or 'holdout'."
            )
        return normalized

    @field_validator("created_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError("created_at must be a datetime instance.")
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be a timezone-aware datetime.")
        return value
