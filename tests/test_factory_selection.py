"""
Factory selection tests for embedding providers.
Tests env vars, routing, and A/B dual-write functionality.
"""
import pytest
import os
from typing import List
from unittest.mock import Mock, patch
from integrations.embeddings_interfaces import EmbeddingProvider


class _MockOpenAIProvider(EmbeddingProvider):
    """Mock OpenAI provider for testing."""
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[10.0] for _ in texts]
    
    def embed_text(self, text: str) -> List[float]:
        return [10.0]
    
    def get_embedding_dimension(self) -> int:
        return 1536


class _MockVoyageProvider(EmbeddingProvider):
    """Mock Voyage provider for testing."""
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[20.0] for _ in texts]
    
    def embed_text(self, text: str) -> List[float]:
        return [20.0]
    
    def get_embedding_dimension(self) -> int:
        return 1024


class _MockCohereProvider(EmbeddingProvider):
    """Mock Cohere provider for testing."""  
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[30.0] for _ in texts]
    
    def embed_text(self, text: str) -> List[float]:
        return [30.0]
    
    def get_embedding_dimension(self) -> int:
        return 1024


@pytest.mark.unit
class TestFactorySelection:
    """Test factory provider selection via environment variables."""
    
    def test_default_provider(self, monkeypatch):
        """Test default provider selection (OpenAI)."""
        # Clean env and set API key
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            mock_create.return_value = _MockOpenAIProvider()
            
            from integrations.embeddings_factory import get_embedder
            embedder = get_embedder()
            result = embedder.embed(["test"])
            assert result == [[10.0]]
            mock_create.assert_called_once_with("openai", None)
    
    def test_env_provider_selection(self, monkeypatch):
        """Test provider selection via EMBEDDING_PROVIDER env var."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            mock_create.return_value = _MockVoyageProvider()
            
            from integrations.embeddings_factory import get_embedder
            embedder = get_embedder()
            result = embedder.embed(["test"])
            assert result == [[20.0]]
            mock_create.assert_called_once_with("voyage", None)
    
    def test_routing_configuration(self, monkeypatch):
        """Test tag-based routing via EMBED_ROUTING."""
        monkeypatch.setenv("EMBED_ROUTING", "obsidian:openai,commits:voyage,default:openai")
        
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            def side_effect(provider, config=None):
                if provider == "openai":
                    return _MockOpenAIProvider()
                elif provider == "voyage":
                    return _MockVoyageProvider()
                return _MockOpenAIProvider()
            
            mock_create.side_effect = side_effect
            
            from integrations.embeddings_factory import get_embedder_for
            
            # Test specific tag routing
            obsidian_embedder = get_embedder_for("obsidian")
            assert obsidian_embedder.embed(["test"]) == [[10.0]]  # OpenAI
            
            commits_embedder = get_embedder_for("commits")
            assert commits_embedder.embed(["test"]) == [[20.0]]  # Voyage
            
            # Test default routing
            other_embedder = get_embedder_for("unknown_tag")
            assert other_embedder.embed(["test"]) == [[10.0]]  # Default to OpenAI
            
            # Test case insensitive
            caps_embedder = get_embedder_for("OBSIDIAN")
            assert caps_embedder.embed(["test"]) == [[10.0]]  # OpenAI
    
    def test_routing_no_default(self, monkeypatch):
        """Test routing when no default is specified."""
        monkeypatch.setenv("EMBED_ROUTING", "obsidian:openai,commits:voyage")  # No default
        
        with patch("integrations.embeddings_factory.get_embedder") as mock_get_embedder:
            mock_get_embedder.return_value = _MockOpenAIProvider()
            
            from integrations.embeddings_factory import get_embedder_for
            
            # Unknown tag should fall back to standard get_embedder
            embedder = get_embedder_for("unknown")
            mock_get_embedder.assert_called_once()
    
    def test_dual_write_comma_format(self, monkeypatch):
        """Test dual-write A/B testing with 'primary,shadow' format."""
        monkeypatch.setenv("EMBED_AB_WRITE", "openai,voyage")
        
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            def side_effect(provider, config=None):
                if provider == "openai":
                    return _MockOpenAIProvider()
                elif provider == "voyage":
                    return _MockVoyageProvider()
                return _MockOpenAIProvider()
            
            mock_create.side_effect = side_effect
            
            with patch("integrations.providers.dualwrite.DualWriteEmbeddings") as mock_dual:
                mock_dual_instance = Mock()
                mock_dual_instance.embed.return_value = [[10.0]]  # Return primary result
                mock_dual.return_value = mock_dual_instance
                
                from integrations.embeddings_factory import get_embedder_ab
                embedder = get_embedder_ab()
                result = embedder.embed(["test"])
                
                # Should create DualWriteEmbeddings with 100% shadow
                mock_dual.assert_called_once()
                call_kwargs = mock_dual.call_args.kwargs
                assert call_kwargs["shadow_percentage"] == 100.0
                assert call_kwargs["compare_results"] is True
                assert result == [[10.0]]
    
    def test_dual_write_percentage_format(self, monkeypatch):
        """Test dual-write A/B testing with 'shadow:percentage' format."""
        monkeypatch.setenv("EMBED_AB_WRITE", "voyage:25")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")  # Primary
        
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            def side_effect(provider, config=None):
                if provider == "openai":
                    return _MockOpenAIProvider()
                elif provider == "voyage":
                    return _MockVoyageProvider()
                return _MockOpenAIProvider()
            
            mock_create.side_effect = side_effect
            
            with patch("integrations.providers.dualwrite.DualWriteEmbeddings") as mock_dual:
                mock_dual_instance = Mock()
                mock_dual_instance.embed.return_value = [[10.0]]
                mock_dual.return_value = mock_dual_instance
                
                from integrations.embeddings_factory import get_embedder_ab
                embedder = get_embedder_ab()
                result = embedder.embed(["test"])
                
                # Should use openai as primary and voyage as shadow with 25%
                mock_dual.assert_called_once()
                call_kwargs = mock_dual.call_args.kwargs
                assert call_kwargs["shadow_percentage"] == 25.0
                assert call_kwargs["compare_results"] is True
                assert result == [[10.0]]
    
    def test_dual_write_no_config(self, monkeypatch):
        """Test get_embedder_ab returns standard embedder when no A/B config."""
        # No EMBED_AB_WRITE set
        
        with patch("integrations.embeddings_factory.get_embedder") as mock_get:
            mock_get.return_value = _MockOpenAIProvider()
            
            from integrations.embeddings_factory import get_embedder_ab
            embedder = get_embedder_ab()
            
            # Should fall back to standard factory
            mock_get.assert_called_once()
            assert embedder.embed(["test"]) == [[10.0]]
    
    def test_dual_write_shadow_unavailable(self, monkeypatch):
        """Test dual-write when shadow provider is unavailable."""
        monkeypatch.setenv("EMBED_AB_WRITE", "unavailable_provider:50")
        
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            def side_effect(provider, config=None):
                if provider == "openai":
                    return _MockOpenAIProvider()
                else:
                    raise ImportError("Provider not available")
            
            mock_create.side_effect = side_effect
            
            from integrations.embeddings_factory import get_embedder_ab
            embedder = get_embedder_ab()
            
            # Should return primary only when shadow fails
            assert embedder.embed(["test"]) == [[10.0]]
            assert isinstance(embedder, _MockOpenAIProvider)
    
    def test_available_providers(self):
        """Test get_available_providers returns correct status."""
        from integrations.embeddings_factory import get_available_providers
        
        providers = get_available_providers()
        
        # OpenAI should always be available
        assert providers["openai"] is True
        
        # DualWrite should always be available (built-in)
        assert providers["dualwrite"] is True
        
        # Others depend on package installation
        assert "voyage" in providers
        assert "cohere" in providers
        assert "fallback" in providers
    
    @pytest.mark.parametrize("provider_name,expected_class", [
        ("openai", "OpenAIEmbeddings"),
        ("voyage", "VoyageEmbeddings"),
        ("cohere", "CohereEmbeddings"),
    ])
    def test_provider_creation(self, provider_name, expected_class, monkeypatch):
        """Test individual provider creation."""
        monkeypatch.setenv(f"{provider_name.upper()}_API_KEY", "test-key")
        
        try:
            from integrations.embeddings_factory import _create_provider
            provider = _create_provider(provider_name)
            assert expected_class in str(type(provider))
        except (ImportError, ValueError):
            # Some providers may not be available in test environment
            pytest.skip(f"{provider_name} provider not available")
    
    def test_config_parameter_passing(self, monkeypatch):
        """Test that config parameters are passed correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            mock_create.return_value = _MockOpenAIProvider()
            
            from integrations.embeddings_factory import get_embedder
            
            config = {"model": "text-embedding-3-large"}
            embedder = get_embedder(config=config)
            
            mock_create.assert_called_once_with("openai", config)