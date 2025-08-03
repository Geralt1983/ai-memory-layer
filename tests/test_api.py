import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from api.main import app, get_memory_engine, get_openai_integration
from core.memory_engine import MemoryEngine, Memory
from tests.conftest import MockEmbeddingProvider


class TestAPI:
    """Test cases for the FastAPI endpoints"""

    @pytest.fixture
    def mock_memory_engine(self):
        """Mock memory engine for testing"""
        engine = Mock(spec=MemoryEngine)
        engine.memories = []
        engine.vector_store = Mock()
        engine.vector_store.__class__.__name__ = "FaissVectorStore"
        engine.vector_store.index.ntotal = 0
        return engine

    @pytest.fixture
    def mock_openai_integration(self):
        """Mock OpenAI integration for testing"""
        integration = Mock()
        integration.chat_with_memory.return_value = "Mock AI response"
        integration.context_builder.build_context.return_value = "Mock context"
        return integration

    @pytest.fixture
    def client(self, mock_memory_engine, mock_openai_integration):
        """Test client with mocked dependencies"""
        app.dependency_overrides[get_memory_engine] = lambda: mock_memory_engine
        app.dependency_overrides[get_openai_integration] = (
            lambda: mock_openai_integration
        )

        client = TestClient(app)
        yield client

        # Cleanup
        app.dependency_overrides.clear()

    def test_health_check(self, client, mock_memory_engine):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["memory_count"] == 0
        assert data["vector_store_type"] == "FaissVectorStore"

    def test_create_memory(self, client, mock_memory_engine):
        """Test creating a new memory"""
        # Setup mock
        mock_memory = Memory(content="Test memory", metadata={"type": "test"})
        mock_memory_engine.add_memory.return_value = mock_memory

        payload = {"content": "Test memory", "metadata": {"type": "test"}}

        response = client.post("/memories", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["content"] == "Test memory"
        assert data["metadata"] == {"type": "test"}
        mock_memory_engine.add_memory.assert_called_once_with(
            "Test memory", {"type": "test"}
        )

    def test_create_memory_validation_error(self, client):
        """Test memory creation with invalid data"""
        payload = {
            "content": "",  # Empty content should fail validation
            "metadata": {"type": "test"},
        }

        response = client.post("/memories", json=payload)
        assert response.status_code == 422  # Validation error

    def test_get_recent_memories(self, client, mock_memory_engine):
        """Test getting recent memories"""
        # Setup mock
        mock_memories = [
            Memory(content="Memory 1", metadata={"type": "test"}),
            Memory(content="Memory 2", metadata={"type": "test"}),
        ]
        mock_memory_engine.get_recent_memories.return_value = mock_memories

        response = client.get("/memories?n=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["content"] == "Memory 1"
        assert data[1]["content"] == "Memory 2"
        mock_memory_engine.get_recent_memories.assert_called_once_with(2)

    def test_get_recent_memories_invalid_n(self, client):
        """Test getting recent memories with invalid n parameter"""
        response = client.get("/memories?n=101")  # Too large
        assert response.status_code == 400

        response = client.get("/memories?n=0")  # Too small
        assert response.status_code == 400

    def test_search_memories(self, client, mock_memory_engine):
        """Test memory search endpoint"""
        # Setup mock
        mock_memories = [
            Memory(
                content="Python programming",
                metadata={"type": "code"},
                relevance_score=0.9,
            )
        ]
        mock_memory_engine.search_memories.return_value = mock_memories

        payload = {"query": "programming", "k": 5}

        response = client.post("/memories/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert len(data["memories"]) == 1
        assert data["memories"][0]["content"] == "Python programming"
        assert data["memories"][0]["relevance_score"] == 0.9
        mock_memory_engine.search_memories.assert_called_once_with("programming", k=5)

    def test_search_memories_validation_error(self, client):
        """Test search with invalid data"""
        payload = {"query": "", "k": 5}  # Empty query should fail

        response = client.post("/memories/search", json=payload)
        assert response.status_code == 422

    def test_clear_memories(self, client, mock_memory_engine):
        """Test clearing all memories"""
        response = client.delete("/memories")

        assert response.status_code == 204
        mock_memory_engine.clear_memories.assert_called_once()

    def test_chat_with_memory(self, client, mock_openai_integration):
        """Test chat endpoint"""
        payload = {
            "message": "Hello AI",
            "system_prompt": "You are helpful",
            "include_recent": 3,
            "include_relevant": 3,
            "remember_response": True,
        }

        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Mock AI response"
        assert data["context_used"] == "Mock context"

        mock_openai_integration.chat_with_memory.assert_called_once_with(
            message="Hello AI",
            system_prompt="You are helpful",
            include_recent=3,
            include_relevant=3,
            remember_response=True,
        )

    def test_chat_validation_error(self, client):
        """Test chat with invalid data"""
        payload = {
            "message": "",  # Empty message should fail
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 422

    def test_chat_with_defaults(self, client, mock_openai_integration):
        """Test chat with default parameters"""
        payload = {"message": "Hello AI"}

        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        mock_openai_integration.chat_with_memory.assert_called_once_with(
            message="Hello AI",
            system_prompt=None,
            include_recent=5,  # Default
            include_relevant=5,  # Default
            remember_response=True,  # Default
        )

    def test_get_stats(self, client, mock_memory_engine):
        """Test stats endpoint"""
        # Setup mock memories with different types
        mock_memories = [
            Memory(content="Memory 1", metadata={"type": "user"}),
            Memory(content="Memory 2", metadata={"type": "system"}),
            Memory(content="Memory 3", metadata={"type": "user"}),
            Memory(content="Memory 4", metadata={}),  # No type
        ]
        mock_memory_engine.memories = mock_memories
        mock_memory_engine.vector_store.index.ntotal = 4

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_memories"] == 4
        assert data["vector_store_entries"] == 4
        assert data["memory_types"]["user"] == 2
        assert data["memory_types"]["system"] == 1
        assert data["memory_types"]["unknown"] == 1
        assert "oldest_memory" in data
        assert "newest_memory" in data

    def test_stats_empty(self, client, mock_memory_engine):
        """Test stats with empty memory"""
        mock_memory_engine.memories = []
        mock_memory_engine.vector_store.index.ntotal = 0

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_memories"] == 0
        assert data["vector_store_entries"] == 0
        assert data["memory_types"] == {}
        assert data["oldest_memory"] is None
        assert data["newest_memory"] is None

    def test_cors_headers(self, client):
        """Test that CORS headers are present"""
        response = client.options("/health")

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


class TestAPIModels:
    """Test Pydantic models"""

    def test_memory_create_model(self):
        """Test MemoryCreate model validation"""
        from api.models import MemoryCreate

        # Valid data
        data = MemoryCreate(content="Test", metadata={"key": "value"})
        assert data.content == "Test"
        assert data.metadata == {"key": "value"}

        # Test with minimal data
        data = MemoryCreate(content="Test")
        assert data.metadata == {}

    def test_chat_request_model(self):
        """Test ChatRequest model validation"""
        from api.models import ChatRequest

        # Valid data with defaults
        data = ChatRequest(message="Hello")
        assert data.message == "Hello"
        assert data.include_recent == 5
        assert data.include_relevant == 5
        assert data.remember_response is True

        # Valid data with custom values
        data = ChatRequest(
            message="Hello",
            system_prompt="Be helpful",
            include_recent=10,
            include_relevant=8,
            remember_response=False,
        )
        assert data.message == "Hello"
        assert data.system_prompt == "Be helpful"
        assert data.include_recent == 10
        assert data.include_relevant == 8
        assert data.remember_response is False

    def test_search_request_model(self):
        """Test SearchRequest model validation"""
        from api.models import SearchRequest

        # Valid data with defaults
        data = SearchRequest(query="test")
        assert data.query == "test"
        assert data.k == 5

        # Valid data with custom k
        data = SearchRequest(query="test", k=10)
        assert data.k == 10
