"""Health check route module."""

from fastapi import APIRouter
from pydantic import BaseModel

try:
    from backend.app.config import get_settings
except ImportError:
    from app.config import get_settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Schema for health status response."""

    status: str
    environment: str
    version: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns the operational status of the FastAPI web service.",
)
def get_health() -> HealthResponse:
    """Return the operational status of the FastAPI service.

    Confirms that the web application is running and responding to requests.
    External dependencies (such as vector stores, embeddings, and LLM APIs)
    are not verified here as they are planned for subsequent phases.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        version=settings.version,
    )
