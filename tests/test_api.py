from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_memory_engine


@pytest.fixture
def client(monkeypatch):
    """Test client with mocked memory engine."""
    engine = Mock()
    engine.get_recent_memories.return_value = []
    engine.memories = []
    engine.vector_store = Mock()
    engine.vector_store.__class__.__name__ = "FaissVectorStore"
    app.dependency_overrides[get_memory_engine] = lambda: engine
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_protected_endpoint_requires_token(client, monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "secret")

    # Missing token
    response = client.get("/memories")
    assert response.status_code == 401

    # Valid token
    headers = {"Authorization": "Bearer secret"}
    response = client.get("/memories", headers=headers)
    assert response.status_code == 200
