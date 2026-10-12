"""Tests for the health check endpoint."""

from fastapi.testclient import TestClient


def test_health_check_returns_ok(client: TestClient) -> None:
    """Test that GET /health returns HTTP 200 and valid status payload."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["environment"] == "development"
    assert data["version"] == "0.1.0"


def test_health_check_post_method_not_allowed(client: TestClient) -> None:
    """Test that POST /health returns HTTP 405 Method Not Allowed."""
    response = client.post("/health")
    assert response.status_code == 405


def test_nonexistent_route_returns_404(client: TestClient) -> None:
    """Test that an unmapped route returns HTTP 404 Not Found."""
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404
