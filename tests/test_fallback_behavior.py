"""
Fallback behavior tests for embedding providers.
Tests that fallback providers correctly handle primary failures.
"""
import pytest
from typing import List
from integrations.embeddings_interfaces import EmbeddingProvider


class _MockPrimaryProvider(EmbeddingProvider):
    """Mock primary provider that can be configured to fail."""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.call_count = 0
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("Primary provider failed")
        return [[10.0 + i] for i in range(len(texts))]
    
    def embed_text(self, text: str) -> List[float]:
        result = self.embed([text])
        return result[0] if result else None
    
    def get_embedding_dimension(self) -> int:
        return 1536


class _MockBackupProvider(EmbeddingProvider):
    """Mock backup provider that should succeed."""
    
    def __init__(self):
        self.call_count = 0
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        return [[20.0 + i] for i in range(len(texts))]
    
    def embed_text(self, text: str) -> List[float]:
        result = self.embed([text])
        return result[0] if result else None
    
    def get_embedding_dimension(self) -> int:
        return 1024


@pytest.mark.unit  
class TestFallbackBehavior:
    """Test fallback provider behavior when primary fails."""
    
    def test_fallback_on_primary_failure(self):
        """Test fallback provider is used when primary fails."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        primary = _MockPrimaryProvider(should_fail=True)
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        result = fallback_embedder.embed(["test1", "test2", "test3"])
        
        # Should get backup results since primary failed
        expected = [[20.0], [21.0], [22.0]]
        assert result == expected
        
        # Primary should have been tried
        assert primary.call_count == 1
        
        # Backup should have been used
        assert backup.call_count == 1
    
    def test_primary_success_no_fallback(self):
        """Test backup is not called when primary succeeds."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        primary = _MockPrimaryProvider(should_fail=False)
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        result = fallback_embedder.embed(["test1", "test2"])
        
        # Should get primary results
        expected = [[10.0], [11.0]]
        assert result == expected
        
        # Primary should have been called
        assert primary.call_count == 1
        
        # Backup should NOT have been called
        assert backup.call_count == 0
    
    def test_fallback_statistics(self):
        """Test fallback statistics tracking."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        primary = _MockPrimaryProvider(should_fail=True)
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        # Make several requests with failures
        fallback_embedder.embed(["test1"])
        fallback_embedder.embed(["test2"])
        
        stats = fallback_embedder.get_stats()
        
        assert stats["primary_failures"] == 2
        assert stats["backup_uses"] == 2
        assert "primary_provider" in stats
        assert "backup_provider" in stats
    
    def test_embed_text_fallback(self):
        """Test fallback for single text embedding."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        primary = _MockPrimaryProvider(should_fail=True)
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        result = fallback_embedder.embed_text("single test")
        
        # Should get backup result
        assert result == [20.0]
        assert primary.call_count == 1
        assert backup.call_count == 1
    
    def test_dimension_from_primary(self):
        """Test dimension is taken from primary provider."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        primary = _MockPrimaryProvider(should_fail=False)  # Success
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        dimension = fallback_embedder.get_embedding_dimension()
        assert dimension == 1536  # Primary dimension
    
    def test_both_providers_fail(self):
        """Test behavior when both primary and backup fail."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        class _FailingBackup(_MockBackupProvider):
            def embed(self, texts: List[str]) -> List[List[float]]:
                self.call_count += 1
                raise RuntimeError("Backup also failed")
        
        primary = _MockPrimaryProvider(should_fail=True)
        backup = _FailingBackup()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        # Both providers fail, should raise the backup exception
        with pytest.raises(RuntimeError, match="Backup also failed"):
            fallback_embedder.embed(["test"])
        
        assert primary.call_count == 1
        assert backup.call_count == 1
    
    def test_empty_input_fallback(self):
        """Test fallback with empty input."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        primary = _MockPrimaryProvider(should_fail=True)
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        result = fallback_embedder.embed([])
        assert result == []
        
        # Should try primary first, then backup
        assert primary.call_count == 1
        assert backup.call_count == 1
    
    def test_mixed_success_failure(self):
        """Test alternating success and failure patterns."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        class _IntermittentPrimary(_MockPrimaryProvider):
            def __init__(self):
                super().__init__(should_fail=False)
                self.failure_count = 0
            
            def embed(self, texts: List[str]) -> List[List[float]]:
                self.call_count += 1
                # Fail every other call
                if self.call_count % 2 == 0:
                    self.failure_count += 1
                    raise RuntimeError("Intermittent failure")
                return super().embed(texts)
        
        primary = _IntermittentPrimary()
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        # First call should succeed (primary)
        result1 = fallback_embedder.embed(["test1"])
        assert result1 == [[10.0]]
        assert backup.call_count == 0
        
        # Second call should fail to backup
        result2 = fallback_embedder.embed(["test2"])
        assert result2 == [[20.0]]  
        assert backup.call_count == 1
        
        # Third call should succeed (primary again)
        result3 = fallback_embedder.embed(["test3"])
        assert result3 == [[10.0]]
        assert backup.call_count == 1  # Still 1, not called again
        
        stats = fallback_embedder.get_stats()
        assert stats["primary_failures"] == 1
        assert stats["backup_uses"] == 1
    
    def test_fallback_reset_stats(self):
        """Test resetting fallback statistics."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        primary = _MockPrimaryProvider(should_fail=True)
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        # Generate some stats
        fallback_embedder.embed(["test"])
        stats_before = fallback_embedder.get_stats()
        assert stats_before["backup_uses"] == 1
        
        # Reset stats
        fallback_embedder.reset_stats()
        stats_after = fallback_embedder.get_stats()
        assert stats_after["backup_uses"] == 0
        assert stats_after["primary_failures"] == 0


@pytest.mark.property
class TestFallbackProperty:
    """Property-based and chaos tests for fallback behavior."""
    
    @pytest.mark.flaky(reruns=2)
    def test_fallback_under_chaos(self):
        """Test fallback behavior under chaotic conditions."""
        import random
        from integrations.providers.fallback import FallbackEmbeddings
        
        class _ChaoticPrimary(_MockPrimaryProvider):
            def __init__(self, failure_rate=0.5):
                super().__init__(should_fail=False)
                self.failure_rate = failure_rate
            
            def embed(self, texts: List[str]) -> List[List[float]]:
                self.call_count += 1
                if random.random() < self.failure_rate:
                    raise RuntimeError("Chaotic failure")
                return super().embed(texts)
        
        primary = _ChaoticPrimary(failure_rate=0.8)  # 80% failure rate
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        # Make many requests - all should eventually succeed via fallback
        results = []
        for i in range(20):
            result = fallback_embedder.embed([f"test_{i}"])
            results.append(result)
        
        # All requests should have succeeded (via primary or backup)
        assert len(results) == 20
        assert all(len(result) == 1 for result in results)
        
        stats = fallback_embedder.get_stats()
        assert stats["success_rate"] == 1.0  # All should succeed
        assert stats["backup_uses"] > 0  # Some should have used backup
    
    @pytest.mark.parametrize("batch_size", [0, 1, 5, 10, 50])
    def test_fallback_variable_batch_sizes(self, batch_size):
        """Test fallback with variable batch sizes."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        primary = _MockPrimaryProvider(should_fail=True)
        backup = _MockBackupProvider()
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        texts = [f"text_{i}" for i in range(batch_size)]
        result = fallback_embedder.embed(texts)
        
        assert len(result) == batch_size
        if batch_size > 0:
            # Should get backup results (20.0 + index)
            expected = [[20.0 + i] for i in range(batch_size)]
            assert result == expected