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
