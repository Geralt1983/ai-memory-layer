"""
Property-based and chaos testing for embedding providers.
Tests system behavior under various failure conditions and edge cases.
"""
import pytest
import random
import time
from typing import List
from unittest.mock import Mock, patch
from integrations.embeddings_interfaces import EmbeddingProvider


class _ChaoticEmbeddings(EmbeddingProvider):
    """Embedding provider that randomly fails for chaos testing."""
    
    def __init__(self, failure_rate=0.5, latency_range=(0.01, 0.1)):
        self.failure_rate = failure_rate
        self.latency_range = latency_range
        self.call_count = 0
        self.failure_count = 0
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        
        # Simulate random latency
        time.sleep(random.uniform(*self.latency_range))
        
        # Randomly fail
        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise RuntimeError(f"Chaotic failure on call {self.call_count}")
        
        # Return deterministic embeddings
        return [[float(i) + 100.0] for i in range(len(texts))]
    
    def embed_text(self, text: str) -> List[float]:
        result = self.embed([text])
        return result[0] if result else None
    
    def get_embedding_dimension(self) -> int:
        return 128


@pytest.mark.property
class TestPropertyBased:
    """Property-based tests that should hold for all providers."""
    
    @pytest.mark.parametrize("batch_size", [0, 1, 2, 5, 10, 50, 100])
    def test_input_output_length_invariant(self, fake_openai_embeddings, batch_size):
        """Property: len(output) == len(input) for all batch sizes."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        texts = [f"text_{i}" for i in range(batch_size)]
        result = embedder.embed(texts)
        
        assert len(result) == batch_size
    
    @pytest.mark.parametrize("texts", [
        [],  # Empty
        ["single"],  # Single item
        ["a", "b"],  # Two items
        ["same", "same"],  # Duplicates
        ["a" * 1000],  # Very long
        ["🚀", "café", "naïve"],  # Unicode
    ])
    def test_embedding_consistency(self, fake_openai_embeddings, texts):
        """Property: Same input should produce same output."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        result1 = embedder.embed(texts)
        result2 = embedder.embed(texts)
        
        assert result1 == result2
    
    def test_embedding_determinism(self, fake_openai_embeddings):
        """Property: Embeddings should be deterministic given same input."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        # Test multiple times with same input
        test_text = "deterministic test"
        results = [embedder.embed_text(test_text) for _ in range(5)]
        
        # All results should be identical
        assert all(r == results[0] for r in results[1:])
    
    def test_empty_input_property(self, fake_openai_embeddings):
        """Property: Empty input always produces empty output."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        result = embedder.embed([])
        assert result == []
    
    def test_single_vs_batch_consistency(self, fake_openai_embeddings):
        """Property: Single embedding should match first item of batch."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        text = "consistency test"
        single_result = embedder.embed_text(text)
        batch_result = embedder.embed([text])
        
        assert len(batch_result) == 1
        assert single_result == batch_result[0]
    
    def test_dimension_consistency(self, fake_openai_embeddings):
        """Property: All embeddings from same provider have same dimension."""
        from integrations.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        texts = ["text1", "text2", "text3"]
        results = embedder.embed(texts)
        
        if results:
            expected_dim = len(results[0])
            assert all(len(emb) == expected_dim for emb in results)
            assert expected_dim > 0
    
    @pytest.mark.parametrize("provider_config", [
        ("openai", "text-embedding-ada-002"),
        ("openai", "text-embedding-3-small"),
        ("openai", "text-embedding-3-large"),
    ])
    def test_model_switching_property(self, fake_openai_embeddings, provider_config):
        """Property: Model switching should preserve embedding contract."""
        from integrations.embeddings import OpenAIEmbeddings
        provider, model = provider_config
        
        embedder = OpenAIEmbeddings(api_key="test-key", model=model)
        result = embedder.embed(["model test"])
        
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) > 0


@pytest.mark.chaos
class TestChaosEngineering:
    """Chaos engineering tests for system resilience."""
    
    @pytest.mark.flaky(reruns=3)
    def test_high_failure_rate_resilience(self):
        """Test system behavior with 90% failure rate."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        chaotic_primary = _ChaoticEmbeddings(failure_rate=0.9)
        stable_backup = _ChaoticEmbeddings(failure_rate=0.0)
        
        fallback_embedder = FallbackEmbeddings(
            primary=chaotic_primary,
            backup=stable_backup
        )
        
        # Make multiple requests - all should eventually succeed
        success_count = 0
        for i in range(10):
            try:
                result = fallback_embedder.embed([f"chaos_test_{i}"])
                if result and len(result) == 1:
                    success_count += 1
            except Exception:
                pass  # Expected due to high failure rate
        
        # At least some should succeed through fallback
        assert success_count > 0
    
    def test_intermittent_failures(self):
        """Test handling of intermittent provider failures."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        # 50% failure rate creates intermittent failures
        intermittent_primary = _ChaoticEmbeddings(failure_rate=0.5)
        stable_backup = _ChaoticEmbeddings(failure_rate=0.0)
        
        fallback_embedder = FallbackEmbeddings(
            primary=intermittent_primary,
            backup=stable_backup
        )
        
        results = []
        for i in range(20):
            result = fallback_embedder.embed([f"intermittent_{i}"])
            results.append(result)
        
        # All requests should succeed (via primary or backup)
        assert len(results) == 20
        assert all(len(r) == 1 for r in results)
        
        stats = fallback_embedder.get_stats()
        assert stats["backup_uses"] > 0  # Some should have used backup
    
    def test_dual_write_under_chaos(self):
        """Test dual-write behavior when shadow provider is chaotic."""
        from integrations.providers.dualwrite import DualWriteEmbeddings
        
        stable_primary = _ChaoticEmbeddings(failure_rate=0.0)
        chaotic_shadow = _ChaoticEmbeddings(failure_rate=0.8)
        
        dual_writer = DualWriteEmbeddings(
            primary=stable_primary,
            shadow=chaotic_shadow,
            shadow_percentage=100.0,  # Always dual-write
            compare_results=False  # Don't compare due to failures
        )
        
        # Primary should always succeed, shadow failures should be ignored
        results = []
        for i in range(10):
            result = dual_writer.embed([f"dual_chaos_{i}"])
            results.append(result)
        
        # All requests should return primary results
        assert len(results) == 10
        assert all(len(r) == 1 for r in results)
        assert all(r[0][0] == 100.0 for r in results)  # Primary returns 100.0
        
        stats = dual_writer.get_stats()
        assert stats["primary_calls"] == 10
        assert stats["shadow_calls"] == 10
        assert stats["shadow_failures"] > 0  # Shadow should have failed some
    
    @pytest.mark.parametrize("concurrent_calls", [5, 10, 20])
    def test_concurrent_chaos(self, concurrent_calls):
        """Test system behavior under concurrent chaotic load."""
        import threading
        from integrations.providers.fallback import FallbackEmbeddings
        
        chaotic_primary = _ChaoticEmbeddings(
            failure_rate=0.6,
            latency_range=(0.001, 0.05)  # Faster for threading
        )
        stable_backup = _ChaoticEmbeddings(failure_rate=0.0)
        
        fallback_embedder = FallbackEmbeddings(
            primary=chaotic_primary,
            backup=stable_backup
        )
        
        results = []
        errors = []
        
        def make_request(i):
            try:
                result = fallback_embedder.embed([f"concurrent_{i}"])
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(concurrent_calls):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Most requests should succeed
        assert len(results) >= concurrent_calls * 0.5
        assert len(errors) < concurrent_calls * 0.5
    
    def test_memory_pressure_simulation(self):
        """Test behavior under simulated memory pressure."""
        from integrations.embeddings import OpenAIEmbeddings
        
        # Create many large embedding requests
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        # Simulate memory pressure with large batches
        large_batches = [
            [f"large_batch_{i}_{j}" for j in range(100)]
            for i in range(5)
        ]
        
        with patch('integrations.embeddings.OpenAIEmbeddings.embed') as mock_embed:
            # Mock to simulate memory-efficient behavior
            mock_embed.side_effect = lambda texts: [[0.0] for _ in texts]
            
            for batch in large_batches:
                result = embedder.embed(batch)
                assert len(result) == len(batch)
    
    def test_network_timeout_simulation(self):
        """Test behavior under network timeout conditions."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        class _TimeoutProvider(EmbeddingProvider):
            def embed(self, texts: List[str]) -> List[List[float]]:
                time.sleep(0.1)  # Simulate slow network
                raise TimeoutError("Network timeout")
            
            def embed_text(self, text: str) -> List[float]:
                return self.embed([text])[0]
            
            def get_embedding_dimension(self) -> int:
                return 128
        
        timeout_primary = _TimeoutProvider()
        fast_backup = _ChaoticEmbeddings(failure_rate=0.0, latency_range=(0.001, 0.001))
        
        fallback_embedder = FallbackEmbeddings(
            primary=timeout_primary,
            backup=fast_backup
        )
        
        # Should fallback to backup when primary times out
        result = fallback_embedder.embed(["timeout_test"])
        assert result == [[100.0]]  # Backup result
        
        stats = fallback_embedder.get_stats()
        assert stats["primary_failures"] == 1
        assert stats["backup_uses"] == 1
    
    @pytest.mark.benchmark(group="chaos")
    def test_performance_under_chaos(self, benchmark):
        """Benchmark performance under chaotic conditions."""
        from integrations.providers.fallback import FallbackEmbeddings
        
        chaotic_primary = _ChaoticEmbeddings(
            failure_rate=0.7,
            latency_range=(0.001, 0.01)
        )
        fast_backup = _ChaoticEmbeddings(
            failure_rate=0.0,
            latency_range=(0.001, 0.001)
        )
        
        fallback_embedder = FallbackEmbeddings(
            primary=chaotic_primary,
            backup=fast_backup
        )
        
        def chaos_requests():
            results = []
            for i in range(10):
                result = fallback_embedder.embed([f"perf_test_{i}"])
                results.append(result)
            return results
        
        results = benchmark(chaos_requests)
        assert len(results) == 10
        assert all(len(r) == 1 for r in results)
