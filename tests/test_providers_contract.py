"""
Contract tests for embedding providers.
Ensures all providers preserve input order and count.
"""
import pytest
from typing import List
from integrations.embeddings_interfaces import EmbeddingProvider


@pytest.mark.unit
class TestProviderContract:
    """Test that all embedding providers follow the same contract."""
    
    def _check_contract(self, embedder: EmbeddingProvider, test_texts: List[str]):
        """Verify an embedder follows the contract."""
        result = embedder.embed(test_texts)
        
        # Contract: preserve count
        assert len(result) == len(test_texts), f"Expected {len(test_texts)} embeddings, got {len(result)}"
        
        # Contract: preserve order (embeddings should be different for different inputs)
        if len(result) > 1:
            assert result[0] != result[1], "Different inputs should produce different embeddings"
        
        # Contract: embedding dimension consistency
        if result:
            dimension = len(result[0])
            assert all(len(emb) == dimension for emb in result), "All embeddings should have same dimension"
            assert dimension > 0, "Embeddings should not be empty"
    
    def test_openai_contract(self, fake_openai_embeddings):
        """Test OpenAI embeddings contract."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        self._check_contract(embedder, ["hello", "world", "test"])
    
    def test_openai_embed_method(self, fake_openai_embeddings):
        """Test OpenAI embed() method specifically."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        result = embedder.embed(["test1", "test2"])
        assert len(result) == 2
        assert len(result[0]) == 2  # Our mock returns 2D embeddings
        assert result[0] == [0.0, 0.1]  # First item gets index 0
        assert result[1] == [1.0, 1.1]  # Second item gets index 1
    
    def test_openai_embed_text_method(self, fake_openai_embeddings):
        """Test OpenAI embed_text() method."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        result = embedder.embed_text("single test")
        assert result is not None
        # Can be either list or numpy array depending on implementation
        try:
            import numpy as np
            assert isinstance(result, (list, np.ndarray))
        except ImportError:
            assert isinstance(result, list)
        assert len(result) > 0
    
    def test_openai_dimension_method(self, fake_openai_embeddings):
        """Test OpenAI get_embedding_dimension() method."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key", model="text-embedding-ada-002")
        
        dimension = embedder.get_embedding_dimension()
        assert dimension == 1536  # Ada-002 dimension
    
    def test_voyage_contract(self, fake_voyage_embeddings):
        """Test Voyage embeddings contract."""
        from integrations.providers.voyage import VoyageEmbeddings
        embedder = VoyageEmbeddings(model="voyage-3", api_key="test-key")
        
        if embedder.is_available():
            self._check_contract(embedder, ["hello", "voyage", "test"])
        else:
            pytest.skip("Voyage client not available")
    
    def test_voyage_specific_behavior(self, fake_voyage_embeddings):
        """Test Voyage-specific behavior."""
        from integrations.providers.voyage import VoyageEmbeddings
        embedder = VoyageEmbeddings(model="voyage-3", api_key="test-key")
        
        if not embedder.is_available():
            pytest.skip("Voyage client not available")
            
        result = embedder.embed(["test1", "test2"])
        assert len(result) == 2
        assert result[0] == [20.0]  # Our mock returns 20.0 + index
        assert result[1] == [21.0]
    
    def test_cohere_contract(self, fake_cohere_embeddings):
        """Test Cohere embeddings contract."""
        from integrations.providers.cohere import CohereEmbeddings
        embedder = CohereEmbeddings(model="embed-english-v3.0", api_key="test-key")
        
        if embedder.is_available():
            self._check_contract(embedder, ["hello", "cohere", "test"])
        else:
            pytest.skip("Cohere client not available")
    
    def test_cohere_specific_behavior(self, fake_cohere_embeddings):
        """Test Cohere-specific behavior."""
        from integrations.providers.cohere import CohereEmbeddings
        embedder = CohereEmbeddings(model="embed-english-v3.0", api_key="test-key")
        
        if not embedder.is_available():
            pytest.skip("Cohere client not available")
            
        result = embedder.embed(["test1", "test2"])
        assert len(result) == 2 
        assert result[0] == [30.0]  # Our mock returns 30.0 + index
        assert result[1] == [31.0]
    
    @pytest.mark.parametrize("text_count", [0, 1, 5, 10])
    def test_variable_batch_sizes(self, fake_openai_embeddings, text_count):
        """Test embedding providers handle variable batch sizes."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        texts = [f"text_{i}" for i in range(text_count)]
        result = embedder.embed(texts)
        
        assert len(result) == text_count
        if text_count > 0:
            assert all(isinstance(emb, list) for emb in result)
    
    def test_empty_input_handling(self, fake_openai_embeddings):
        """Test that providers handle empty input gracefully."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        result = embedder.embed([])
        assert result == []
    
    def test_unicode_text_handling(self, fake_openai_embeddings):
        """Test that providers handle unicode text."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        unicode_texts = ["Hello 世界", "café", "naïve", "🚀 rocket"]
        result = embedder.embed(unicode_texts)
        
        assert len(result) == len(unicode_texts)
        assert all(isinstance(emb, list) and len(emb) > 0 for emb in result)
    
    def test_long_text_handling(self, fake_openai_embeddings):
        """Test that providers handle long text inputs."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        long_text = "This is a very long text. " * 100  # ~2700 characters
        result = embedder.embed([long_text])
        
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) > 0