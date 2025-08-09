"""Test fallback embedding provider for high availability."""

import pytest
from typing import List, Optional
from unittest.mock import Mock, MagicMock


class TestFallbackEmbeddings:
    """Test the fallback embedding provider."""
    
    def test_fallback_when_primary_fails(self):
        """Test that backup is used when primary fails."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        # Create mock providers
        primary = Mock()
        primary.embed.side_effect = RuntimeError("Primary provider failed")
        
        backup = Mock()
        backup.embed.return_value = [[0.1], [0.2], [0.3]]
        
        # Create fallback provider
        fallback = FallbackEmbeddings(primary=primary, backup=backup, log_failures=False)
        
        # Test fallback
        texts = ["a", "b", "c"]
        vectors = fallback.embed(texts)
        
        # Verify backup was used
        assert vectors == [[0.1], [0.2], [0.3]]
        primary.embed.assert_called_once_with(texts)
        backup.embed.assert_called_once_with(texts)
        
        # Check stats
        stats = fallback.get_stats()
        assert stats["primary_failures"] == 1
        assert stats["backup_uses"] == 1
    
    def test_primary_success_no_fallback(self):
        """Test that backup is not used when primary succeeds."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        # Create mock providers
        primary = Mock()
        primary.embed.return_value = [[1.0], [2.0], [3.0]]
        
        backup = Mock()
        backup.embed.return_value = [[0.1], [0.2], [0.3]]
        
        # Create fallback provider
        fallback = FallbackEmbeddings(primary=primary, backup=backup)
        
        # Test normal operation
        texts = ["a", "b", "c"]
        vectors = fallback.embed(texts)
        
        # Verify primary was used, backup was not
        assert vectors == [[1.0], [2.0], [3.0]]
        primary.embed.assert_called_once_with(texts)
        backup.embed.assert_not_called()
        
        # Check stats
        stats = fallback.get_stats()
        assert stats["primary_failures"] == 0
        assert stats["backup_uses"] == 0
    
    def test_both_providers_fail(self):
        """Test error when both providers fail."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        # Create mock providers that both fail
        primary = Mock()
        primary.embed.side_effect = RuntimeError("Primary failed")
        
        backup = Mock()
        backup.embed.side_effect = RuntimeError("Backup failed")
        
        # Create fallback provider
        fallback = FallbackEmbeddings(primary=primary, backup=backup, log_failures=False)
        
        # Test that exception is raised
        texts = ["a", "b", "c"]
        with pytest.raises(RuntimeError) as exc_info:
            fallback.embed(texts)
        
        assert "Backup failed" in str(exc_info.value)
        
        # Both should have been called
        primary.embed.assert_called_once_with(texts)
        backup.embed.assert_called_once_with(texts)
    
    def test_single_text_fallback(self):
        """Test fallback for single text embedding."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        # Create mock providers
        primary = Mock()
        primary.embed_text.side_effect = Exception("Primary failed")
        
        backup = Mock()
        backup.embed_text.return_value = [0.5, 0.6]
        
        # Create fallback provider
        fallback = FallbackEmbeddings(primary=primary, backup=backup, log_failures=False)
        
        # Test single text fallback
        vector = fallback.embed_text("test text")
        
        assert vector == [0.5, 0.6]
        primary.embed_text.assert_called_once_with("test text")
        backup.embed_text.assert_called_once_with("test text")
    
    def test_dimension_fallback(self):
        """Test getting embedding dimension with fallback."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        # Primary fails to get dimension
        primary = Mock()
        primary.get_embedding_dimension.side_effect = Exception("Primary failed")
        
        backup = Mock()
        backup.get_embedding_dimension.return_value = 768
        
        # Create fallback provider
        fallback = FallbackEmbeddings(primary=primary, backup=backup)
        
        # Get dimension
        dim = fallback.get_embedding_dimension()
        
        assert dim == 768
        primary.get_embedding_dimension.assert_called_once()
        backup.get_embedding_dimension.assert_called_once()
    
    def test_stats_tracking(self):
        """Test that failure statistics are tracked correctly."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        # Create providers with intermittent failures
        primary = Mock()
        primary.embed.side_effect = [
            RuntimeError("Fail 1"),
            [[1.0]],  # Success
            RuntimeError("Fail 2"),
            RuntimeError("Fail 3"),
            [[2.0]],  # Success
        ]
        
        backup = Mock()
        backup.embed.return_value = [[0.5]]
        
        # Create fallback provider
        fallback = FallbackEmbeddings(primary=primary, backup=backup, log_failures=False)
        
        # Perform multiple operations
        fallback.embed(["a"])  # Fail -> backup
        fallback.embed(["b"])  # Success
        fallback.embed(["c"])  # Fail -> backup
        fallback.embed(["d"])  # Fail -> backup
        fallback.embed(["e"])  # Success
        
        # Check statistics
        stats = fallback.get_stats()
        assert stats["primary_failures"] == 3
        assert stats["backup_uses"] == 3
        
        # Reset stats
        fallback.reset_stats()
        stats = fallback.get_stats()
        assert stats["primary_failures"] == 0
        assert stats["backup_uses"] == 0


class TestMultiProviderFallback:
    """Test multi-provider fallback chain."""
    
    def test_multi_provider_chain(self):
        """Test fallback through multiple providers."""
        from integrations.providers.fallback import MultiProviderFallback
        
        # Create mock providers
        provider1 = Mock()
        provider1.embed.side_effect = RuntimeError("Provider 1 failed")
        
        provider2 = Mock()
        provider2.embed.side_effect = RuntimeError("Provider 2 failed")
        
        provider3 = Mock()
        provider3.embed.return_value = [[3.0], [3.1], [3.2]]
        
        provider4 = Mock()
        provider4.embed.return_value = [[4.0], [4.1], [4.2]]
        
        # Create multi-provider fallback
        multi_fallback = MultiProviderFallback(
            providers=[provider1, provider2, provider3, provider4],
            log_failures=False
        )
        
        # Test fallback chain
        texts = ["a", "b", "c"]
        vectors = multi_fallback.embed(texts)
        
        # Should use provider3 (first successful)
        assert vectors == [[3.0], [3.1], [3.2]]
        
        # Check that providers were tried in order
        provider1.embed.assert_called_once_with(texts)
        provider2.embed.assert_called_once_with(texts)
        provider3.embed.assert_called_once_with(texts)
        provider4.embed.assert_not_called()  # Not needed
        
        # Check stats
        stats = multi_fallback.get_stats()
        assert stats["failure_counts"] == [1, 1, 0, 0]
    
    def test_all_providers_fail(self):
        """Test error when all providers in chain fail."""
        from integrations.providers.fallback import MultiProviderFallback
        
        # Create mock providers that all fail
        providers = []
        for i in range(3):
            provider = Mock()
            provider.embed.side_effect = RuntimeError(f"Provider {i} failed")
            provider.__class__.__name__ = f"Provider{i}"
            providers.append(provider)
        
        # Create multi-provider fallback
        multi_fallback = MultiProviderFallback(providers=providers, log_failures=False)
        
        # Test that exception is raised
        texts = ["a", "b", "c"]
        with pytest.raises(RuntimeError) as exc_info:
            multi_fallback.embed(texts)
        
        error_msg = str(exc_info.value)
        assert "All embedding providers failed" in error_msg
        assert "Provider0" in error_msg
        assert "Provider1" in error_msg
        assert "Provider2" in error_msg
    
    def test_empty_provider_list(self):
        """Test that empty provider list raises error."""
        from integrations.providers.fallback import MultiProviderFallback
        
        with pytest.raises(ValueError) as exc_info:
            MultiProviderFallback(providers=[])
        
        assert "At least one provider required" in str(exc_info.value)
    
    def test_single_text_multi_fallback(self):
        """Test single text embedding with multi-provider fallback."""
        from integrations.providers.fallback import MultiProviderFallback
        
        # Create mock providers
        provider1 = Mock()
        provider1.embed_text.return_value = None
        
        provider2 = Mock()
        provider2.embed_text.return_value = [2.0, 2.1]
        
        # Create multi-provider fallback
        multi_fallback = MultiProviderFallback(
            providers=[provider1, provider2],
            log_failures=False
        )
        
        # Test single text
        vector = multi_fallback.embed_text("test")
        
        assert vector == [2.0, 2.1]
        provider1.embed_text.assert_called_once_with("test")
        provider2.embed_text.assert_called_once_with("test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])