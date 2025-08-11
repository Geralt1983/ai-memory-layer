"""Test embedding provider contract - ensures all providers maintain same behavior."""

import os
import pytest
from typing import List
from unittest.mock import Mock, patch, MagicMock


class TestEmbeddingContract:
    """Test that embedding providers follow the contract."""
    
    def test_openai_contract_order_and_length(self, monkeypatch):
        """Ensure OpenAI provider returns one vector per input and preserves order."""
        # Set up environment
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        # Mock OpenAI client
        with patch("integrations.embeddings.openai") as mock_openai:
            # Mock the OpenAI client and its embedding response
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            
            # Create mock response with embeddings
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[float(i), float(i)+0.01])
                for i in range(3)
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            # Import and use factory
            from integrations.embeddings_factory import get_embedder
            embedder = get_embedder()
            
            # Test with multiple texts
            texts = ["text a", "text b", "text c"]
            vectors = embedder.embed(texts)
            
            # Verify contract
            assert len(vectors) == len(texts), "Must return one vector per input"
            assert vectors[0] != vectors[1], "Vectors should be different"
            assert vectors[1] != vectors[2], "Vectors should be different"
            
            # Verify order preserved
            assert vectors[0] == [0.0, 0.01]
            assert vectors[1] == [1.0, 1.01]
            assert vectors[2] == [2.0, 2.01]
    
    def test_single_text_embedding(self, monkeypatch):
        """Test single text embedding convenience method."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        with patch("integrations.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            
            # Mock single embedding response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.5, 0.6, 0.7])]
            mock_client.embeddings.create.return_value = mock_response
            
            from integrations.embeddings_factory import get_embedder
            embedder = get_embedder()
            
            # Test single text
            vector = embedder.embed_text("single text")
            
            assert vector is not None
            assert len(vector) == 3
            assert vector == [0.5, 0.6, 0.7]
    
    def test_empty_input_handling(self, monkeypatch):
        """Test handling of empty input lists."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        with patch("integrations.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            
            from integrations.embeddings_factory import get_embedder
            embedder = get_embedder()
            
            # Test empty list
            vectors = embedder.embed([])
            assert vectors == []
    
    def test_dimension_consistency(self, monkeypatch):
        """Test that all vectors have the same dimension."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        with patch("integrations.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            
            # Create embeddings of consistent dimension
            dimension = 1536
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[float(i % 10) / 10] * dimension)
                for i in range(5)
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            from integrations.embeddings_factory import get_embedder
            embedder = get_embedder()
            
            texts = ["text1", "text2", "text3", "text4", "text5"]
            vectors = embedder.embed(texts)
            
            # All vectors should have same dimension
            dimensions = [len(v) for v in vectors]
            assert all(d == dimension for d in dimensions)
            assert embedder.get_embedding_dimension() == dimension
    
    def test_batch_size_handling(self, monkeypatch):
        """Test that large batches are handled correctly."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        with patch("integrations.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            
            # Test with large batch
            batch_size = 100
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[float(i)])
                for i in range(batch_size)
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            from integrations.embeddings_factory import get_embedder
            embedder = get_embedder()
            
            texts = [f"text_{i}" for i in range(batch_size)]
            vectors = embedder.embed(texts)
            
            assert len(vectors) == batch_size
            # Verify order preserved
            for i in range(batch_size):
                assert vectors[i] == [float(i)]


class TestProviderFactory:
    """Test the embedding provider factory."""
    
    def test_default_provider_is_openai(self, monkeypatch):
        """Test that OpenAI is the default provider."""
        # Remove any existing env var
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        with patch("integrations.embeddings.openai"):
            from integrations.embeddings_factory import get_embedder
            embedder = get_embedder()
            
            # Should be OpenAI embeddings
            assert embedder.__class__.__name__ == "OpenAIEmbeddings"
    
    def test_env_var_provider_selection(self, monkeypatch):
        """Test provider selection via environment variable."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        with patch("integrations.embeddings.openai"):
            from integrations.embeddings_factory import get_embedder
            embedder = get_embedder()
            
            assert embedder.__class__.__name__ == "OpenAIEmbeddings"
    
    def test_unknown_provider_raises_error(self, monkeypatch):
        """Test that unknown provider raises ValueError."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "unknown_provider")
        
        from integrations.embeddings_factory import get_embedder
        
        with pytest.raises(ValueError) as exc_info:
            get_embedder()
        
        assert "Unknown EMBEDDING_PROVIDER" in str(exc_info.value)
    
    def test_available_providers_check(self):
        """Test checking available providers."""
        from integrations.embeddings_factory import get_available_providers
        
        providers = get_available_providers()
        
        # OpenAI should always be available
        assert providers["openai"] is True
        
        # Check structure
        assert "voyage" in providers
        assert "cohere" in providers
        assert "fallback" in providers
        assert "cached" in providers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])