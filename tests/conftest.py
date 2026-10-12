"""Pytest fixtures and test configurations."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI TestClient instance."""
    app = create_app()
    return TestClient(app)
