"""Tests for provider registry and autodiscovery."""

import pytest
from memory_layer.providers import registry as preg
from memory_layer.providers.registry import register
from memory_layer.providers.base import EmbeddingProvider, EmbeddingConfig
from memory_layer.config import AppConfig, build_provider
from typing import List, Sequence


class MockRegisteredProvider(EmbeddingProvider):
    """Mock provider for testing registration."""
    
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__(cfg)
        self.available = True
    
    def is_available(self) -> bool:
        return self.available
    
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.mark.unit
class TestProviderRegistry:
    """Test provider registry functionality."""
    
    def test_builtin_providers_present(self):
        """Test that builtin providers are registered."""
        names = preg.list_providers()
        assert "voyage" in names
        # openai present if module exists locally
        
        # Test that we can get a factory
        factory = preg.get("voyage")
        assert factory is not None
        assert callable(factory)
    
    def test_build_provider_integration(self):
        """Test provider building through AppConfig."""
        cfg = AppConfig(provider="voyage")
        provider = build_provider(cfg)
        assert provider is not None
        assert hasattr(provider, 'is_available')
        assert hasattr(provider, 'embed_batch')
    
    def test_unknown_provider_error(self):
        """Test error handling for unknown providers."""
        cfg = AppConfig(provider="nonexistent-provider")
        with pytest.raises(ValueError, match="Unknown provider"):
            build_provider(cfg)
    
    def test_provider_registration_decorator(self):
        """Test runtime provider registration with decorator."""
        
        @register("test-mock")
        class TestMockProvider(EmbeddingProvider):
            def is_available(self) -> bool:
                return True
            
            def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
                return [[1.0, 2.0] for _ in texts]
        
        # Should now be available in registry
        names = preg.list_providers()
        assert "test-mock" in names
        
        # Should be able to build it
        factory = preg.get("test-mock")
        assert factory is not None
        
        cfg = EmbeddingConfig(model="test", dim=2)
        provider = factory(cfg)
        assert isinstance(provider, TestMockProvider)
        assert provider.is_available()
        
        # Test through AppConfig
        app_cfg = AppConfig(provider="test-mock", model="test", dim=2)
        provider2 = build_provider(app_cfg)
        assert isinstance(provider2, TestMockProvider)
    
    def test_list_providers_sorted(self):
        """Test that provider list is sorted."""
        providers = preg.list_providers()
        assert providers == sorted(providers)
    
    def test_get_nonexistent_provider(self):
        """Test getting nonexistent provider returns None."""
        factory = preg.get("definitely-does-not-exist")
        assert factory is None
    
    def test_load_entrypoints_graceful(self):
        """Test that loading entrypoints doesn't crash on missing groups."""
        # This should not raise an exception even if group doesn't exist
        preg.load_entrypoints("nonexistent.group")
        
        # Standard group should also not crash
        preg.load_entrypoints()
    
    def test_registry_reinitialization(self):
        """Test that built-in providers are available after re-init."""
        # Force re-registration
        preg.register_builtin_providers()
        
        names = preg.list_providers()
        assert "voyage" in names
        
        # Should still work
        factory = preg.get("voyage")
        assert factory is not None


@pytest.mark.integration
class TestProviderIntegration:
    """Integration tests with real provider configs."""
    
    def test_voyage_provider_creation(self):
        """Test creating Voyage provider through registry."""
        factory = preg.get("voyage")
        assert factory is not None
        
        cfg = EmbeddingConfig(model="voyage-2", dim=1536)
        provider = factory(cfg)
        
        assert provider.cfg.model == "voyage-2"
        assert provider.cfg.dim == 1536
        # is_available() depends on VOYAGE_API_KEY
    
    def test_openai_provider_creation_if_available(self):
        """Test creating OpenAI provider if available."""
        factory = preg.get("openai")
        if factory is None:
            pytest.skip("OpenAI provider not available")
        
        cfg = EmbeddingConfig(model="text-embedding-3-small", dim=1536)
        provider = factory(cfg)
        
        assert provider.cfg.model == "text-embedding-3-small"
        assert provider.cfg.dim == 1536
    
    def test_config_environment_integration(self):
        """Test that AppConfig reads from environment correctly."""
        import os
        from unittest.mock import patch
        
        with patch.dict(os.environ, {
            'EMBED_PROVIDER': 'voyage',
            'EMBED_MODEL': 'voyage-3',
            'EMBED_DIM': '1024'
        }):
            cfg = AppConfig()
            assert cfg.provider == 'voyage'
            assert cfg.model == 'voyage-3'
            assert cfg.dim == 1024
            
            # Should be able to build
            provider = build_provider(cfg)
            assert provider.cfg.model == 'voyage-3'
            assert provider.cfg.dim == 1024