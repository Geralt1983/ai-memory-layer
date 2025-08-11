"""
Contract tests for embedding providers.
Ensures all providers preserve input order and count.
Enhanced with new memory_layer provider architecture.
"""
import pytest
from typing import List
from integrations.embeddings_interfaces import EmbeddingProvider
from memory_layer.providers.base import EmbeddingProvider as NewEmbeddingProvider, EmbeddingConfig, ProviderUnavailable


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


@pytest.mark.unit
class TestNewProviderArchitecture:
    """Tests for the new memory_layer provider architecture."""
    
    class MockNewProvider(NewEmbeddingProvider):
        """Mock provider for testing new architecture."""
        
        def __init__(self, cfg: EmbeddingConfig, available: bool = True, should_fail: bool = False):
            super().__init__(cfg)
            self._available = available
            self._should_fail = should_fail
        
        def is_available(self) -> bool:
            return self._available
        
        def embed_batch(self, texts) -> List[List[float]]:
            if not self.is_available():
                raise ProviderUnavailable("Mock provider unavailable")
            if self._should_fail:
                raise RuntimeError("Mock provider failed")
            # Return mock embeddings with correct dimensions
            return [[float(i), 0.1, 0.2] for i in range(len(texts))]
    
    def test_new_provider_contract_compliance(self):
        """Test that new providers implement the required contract."""
        config = EmbeddingConfig(model="test-model", dim=3)
        provider = self.MockNewProvider(config)
        
        assert isinstance(provider.is_available(), bool)
        assert hasattr(provider, 'embed_batch')
        assert hasattr(provider, 'embed_query')
        assert provider.cfg.model == "test-model"
        assert provider.cfg.dim == 3
    
    def test_new_embed_batch_preserves_order(self):
        """Test that new embed_batch returns results in same order as input."""
        config = EmbeddingConfig(model="test-model", dim=3)
        provider = self.MockNewProvider(config)
        
        texts = ["first", "second", "third"]
        results = provider.embed_batch(texts)
        
        assert len(results) == len(texts)
        assert len(results[0]) == config.dim
        # Check ordering: first embedding should have 0.0, second should have 1.0, etc.
        assert results[0][0] == 0.0
        assert results[1][0] == 1.0
        assert results[2][0] == 2.0
    
    def test_new_embed_query_single_result(self):
        """Test that new embed_query returns single vector."""
        config = EmbeddingConfig(model="test-model", dim=3)
        provider = self.MockNewProvider(config)
        
        result = provider.embed_query("test text")
        assert isinstance(result, list)
        assert len(result) == config.dim
        assert all(isinstance(x, (int, float)) for x in result)
    
    def test_new_unavailable_provider_raises(self):
        """Test that unavailable new providers raise appropriate errors."""
        config = EmbeddingConfig(model="test-model", dim=3)
        provider = self.MockNewProvider(config, available=False)
        
        with pytest.raises(ProviderUnavailable):
            provider.embed_batch(["test"])
    
    def test_new_empty_input_handling(self):
        """Test new providers handle empty input gracefully."""
        config = EmbeddingConfig(model="test-model", dim=3)
        provider = self.MockNewProvider(config)
        
        result = provider.embed_batch([])
        assert result == []
    
    def test_new_provider_error_propagation(self):
        """Test that new provider errors are properly propagated."""
        config = EmbeddingConfig(model="test-model", dim=3)
        provider = self.MockNewProvider(config, should_fail=True)
        
        with pytest.raises(RuntimeError, match="Mock provider failed"):
            provider.embed_batch(["test"])
    
    def test_new_configuration_access(self):
        """Test that configuration is accessible through provider."""
        config = EmbeddingConfig(
            model="test-model", 
            dim=512, 
            normalize=False,
            timeout_s=45.0,
            max_batch_size=64,
            max_retries=5
        )
        provider = self.MockNewProvider(config)
        
        assert provider.cfg.model == "test-model"
        assert provider.cfg.dim == 512
        assert provider.cfg.normalize is False
        assert provider.cfg.timeout_s == 45.0
        assert provider.cfg.max_batch_size == 64
        assert provider.cfg.max_retries == 5
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50, 100])
    def test_new_batch_size_handling(self, batch_size):
        """Test that new providers handle various batch sizes."""
        config = EmbeddingConfig(model="test-model", dim=3, max_batch_size=10)
        provider = self.MockNewProvider(config)
        
        texts = [f"text_{i}" for i in range(batch_size)]
        results = provider.embed_batch(texts)
        
        assert len(results) == batch_size
        assert all(len(vec) == 3 for vec in results)
        # Verify ordering is preserved
        for i in range(min(10, batch_size)):  # Check first few
            assert results[i][0] == float(i)
    
    def test_new_provider_stats_method(self):
        """Test that providers can provide stats (if implemented)."""
        config = EmbeddingConfig(model="test-model", dim=3)
        provider = self.MockNewProvider(config)
        
        # Base provider should have stats method (from protocol)
        # Implementation is optional, so we just check it doesn't crash
        try:
            stats = getattr(provider, 'stats', lambda: {})()
            assert isinstance(stats, dict)
        except AttributeError:
            # stats() method is optional in base implementation
            pass


@pytest.mark.integration 
class TestNewVoyageProvider:
    """Integration tests for new Voyage provider architecture."""
    
    def test_voyage_new_architecture(self):
        """Test Voyage provider with new architecture."""
        try:
            from memory_layer.providers.voyage import VoyageEmbeddings
            config = EmbeddingConfig(model="voyage-3", dim=1024, normalize=True)
            provider = VoyageEmbeddings(config)
            
            # Should implement contract
            assert hasattr(provider, 'is_available')
            assert hasattr(provider, 'embed_batch')
            assert hasattr(provider, 'embed_query')
            
            # Availability depends on API key
            available = provider.is_available()
            assert isinstance(available, bool)
            
            if not available:
                # Should raise ProviderUnavailable when trying to embed
                with pytest.raises(ProviderUnavailable):
                    provider.embed_batch(["test"])
            
        except ImportError:
            pytest.skip("New Voyage provider not available")


@pytest.mark.property
class TestNewProviderProperties:
    """Property-based tests for new provider architecture."""
    
    def test_embed_batch_count_invariant(self):
        """Property: len(embed_batch(texts)) == len(texts) for new providers"""
        config = EmbeddingConfig(model="test-model", dim=3)
        provider = TestNewProviderArchitecture.MockNewProvider(config)
        
        for n in [0, 1, 5, 10, 100]:
            texts = [f"text_{i}" for i in range(n)]
            results = provider.embed_batch(texts)
            assert len(results) == n, f"Failed for n={n}"
    
    def test_embed_batch_dimension_invariant(self):
        """Property: all vectors have config.dim dimensions for new providers"""
        config = EmbeddingConfig(model="test-model", dim=5)
        provider = TestNewProviderArchitecture.MockNewProvider(config)
        
        texts = ["short", "medium length text", "very long text with many words"]
        results = provider.embed_batch(texts)
        
        for i, vec in enumerate(results):
            assert len(vec) == 5, f"Vector {i} has wrong dimensions"
    
    def test_embed_query_equivalence(self):
        """Property: embed_query(text) == embed_batch([text])[0] for new providers"""
        config = EmbeddingConfig(model="test-model", dim=3)
        provider = TestNewProviderArchitecture.MockNewProvider(config)
        
        text = "test text"
        single_result = provider.embed_query(text)
        batch_result = provider.embed_batch([text])[0]
        
        assert single_result == batch_result