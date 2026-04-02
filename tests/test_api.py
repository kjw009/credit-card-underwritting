"""Integration tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["feature_count"] == 78


def test_home_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_error_codes(client):
    response = client.get("/error-codes")
    assert response.status_code == 200
    assert "MISSING_FEATURES" in response.json()
