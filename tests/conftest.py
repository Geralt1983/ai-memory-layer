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


# Production-grade test fixtures for comprehensive embedding testing


@pytest.fixture(autouse=True)
def clean_embedding_env(monkeypatch):
    """Clean up embedding-related environment variables before each test."""
    for k in ["EMBEDDING_PROVIDER", "EMBED_ROUTING", "EMBED_AB_WRITE",
              "OPENAI_EMBED_MODEL", "VOYAGE_EMBED_MODEL", "COHERE_EMBED_MODEL"]:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def fake_openai_embeddings(monkeypatch):
    """Mock OpenAI embeddings client to avoid network calls."""
    
    class _MockData:
        def __init__(self, embedding):
            self.embedding = embedding
    
    class _MockResponse:
        def __init__(self, embeddings):
            self.data = [_MockData(emb) for emb in embeddings]
    
    class _MockEmbeddings:
        @staticmethod
        def create(model, input, **kwargs):
            # Return deterministic embeddings based on input
            embeddings = [[float(i), float(i) + 0.1] for i, _ in enumerate(input)]
            return _MockResponse(embeddings)
    
    class _MockClient:
        def __init__(self, api_key=None, **kwargs):
            self.embeddings = _MockEmbeddings()
    
    # Monkeypatch the OpenAI client in our embeddings module
    monkeypatch.setattr("integrations.embeddings.OpenAI", _MockClient)
    yield


@pytest.fixture 
def fake_voyage_embeddings(monkeypatch):
    """Mock Voyage client to avoid network calls."""
    
    class _MockVoyageResponse:
        def __init__(self, texts):
            # Voyage-specific mock embeddings (different from OpenAI)
            self.embeddings = [[20.0 + float(i)] for i, _ in enumerate(texts)]
    
    class _MockVoyageClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
        
        def embed(self, texts, model, input_type=None, truncation=None):
            return _MockVoyageResponse(texts)
    
    # Patch the VoyageEmbeddings class to use mock client
    from integrations.providers.voyage import VoyageEmbeddings
    original_init = VoyageEmbeddings.__init__
    
    def mock_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._set_client_for_tests(_MockVoyageClient())
    
    monkeypatch.setattr(VoyageEmbeddings, "__init__", mock_init)
    yield


@pytest.fixture
def fake_cohere_embeddings(monkeypatch):
    """Mock Cohere client to avoid network calls."""
    
    class _MockCohereResponse:
        def __init__(self, texts):
            # Cohere-specific mock embeddings (different from others)  
            self.embeddings = [[30.0 + float(i)] for i, _ in enumerate(texts)]
    
    class _MockCohereClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
        
        def embed(self, texts, model, input_type=None, truncate=None):
            return _MockCohereResponse(texts)
    
    # Patch the CohereEmbeddings class to use mock client
    from integrations.providers.cohere import CohereEmbeddings
    original_init = CohereEmbeddings.__init__
    
    def mock_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._set_client_for_tests(_MockCohereClient())
    
    monkeypatch.setattr(CohereEmbeddings, "__init__", mock_init)
    yield
