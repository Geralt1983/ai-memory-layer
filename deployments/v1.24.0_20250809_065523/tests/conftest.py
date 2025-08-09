import pytest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, MagicMock
from core.memory_engine import Memory, MemoryEngine
from integrations.embeddings import EmbeddingProvider
from storage.faiss_store import FaissVectorStore
from storage.chroma_store import ChromaVectorStore


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing"""

    def embed_text(self, text):
        # Return a simple hash-based embedding for testing
        return np.array([hash(text) % 1000 for _ in range(10)], dtype="float32")

    def embed_batch(self, texts):
        return [self.embed_text(text) for text in texts]


@pytest.fixture
def mock_embedding_provider():
    """Provides a mock embedding provider"""
    return MockEmbeddingProvider()


@pytest.fixture
def sample_memory():
    """Provides a sample memory for testing"""
    return Memory(
        content="Test memory content",
        metadata={"type": "test", "category": "sample"},
        relevance_score=0.8,
    )


@pytest.fixture
def sample_memories():
    """Provides multiple sample memories for testing"""
    return [
        Memory(content="First memory", metadata={"type": "test"}),
        Memory(content="Second memory", metadata={"type": "user"}),
        Memory(content="Third memory", metadata={"type": "system"}),
    ]


@pytest.fixture
def temp_file():
    """Provides a temporary file path for testing persistence"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_dir():
    """Provides a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_path:
        yield temp_path


@pytest.fixture
def faiss_store():
    """Provides a FAISS vector store for testing"""
    return FaissVectorStore(dimension=10)


@pytest.fixture
def memory_engine(mock_embedding_provider, temp_file):
    """Provides a memory engine with mocked dependencies"""
    faiss_store = FaissVectorStore(dimension=10)
    return MemoryEngine(
        vector_store=faiss_store,
        embedding_provider=mock_embedding_provider,
        persist_path=temp_file,
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()

    # Mock chat completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mock AI response"
    mock_client.chat.completions.create.return_value = mock_response

    # Mock embedding response
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock()]
    mock_embedding_response.data[0].embedding = [0.1] * 1536
    mock_client.embeddings.create.return_value = mock_embedding_response

    return mock_client
