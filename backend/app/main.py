"""FastAPI Application entry point and factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.app.config import get_settings
    from backend.app.api.routes.health import router as health_router
    from backend.app.api.routes.ingestion import router as ingestion_router
    from backend.app.api.routes.retrieval import router as retrieval_router
    from backend.app.api.routes.rag import router as rag_router
except ImportError:
    from app.config import get_settings
    from app.api.routes.health import router as health_router
    from app.api.routes.ingestion import router as ingestion_router
    from app.api.routes.retrieval import router as retrieval_router
    from app.api.routes.rag import router as rag_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description="Backend API service for the RAG Evaluation Harness.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS middleware with environment-driven allowlist
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(health_router)
    app.include_router(ingestion_router)
    app.include_router(retrieval_router)
    app.include_router(rag_router)

    return app


app = create_app()