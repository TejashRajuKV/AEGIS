"""Tests for the /api/health endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_healthy():
    """GET /api/health should return status 'healthy'."""
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "version" in body
