"""Tests for dual-write embedding provider functionality."""

import os
import pytest
from typing import List, Optional
from unittest.mock import Mock, patch

# Test imports
try:
    from integrations.providers.dualwrite import DualWriteEmbeddings
    from integrations.embeddings_factory import get_embedder, get_embedder_ab, _create_dualwrite_provider
    from integrations.embeddings_interfaces import EmbeddingProvider
    from integrations.embeddings import OpenAIEmbeddings
except ImportError as e:
    pytest.skip(f"Embedding modules not available: {e}", allow_module_level=True)


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    
    def __init__(self, name: str, dimension: int = 1536, fail_on_embed: bool = False):
        self.name = name
        self.dimension = dimension
        self.fail_on_embed = fail_on_embed
        self.embed_calls = []
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Mock embed with configurable failure."""
        self.embed_calls.append(texts)
        
        if self.fail_on_embed:
            raise RuntimeError(f"{self.name} embedding failed")
            
        # Return mock embeddings based on provider name for differentiation
        return [[float(ord(c)) for c in self.name[:self.dimension]] + [0.0] * (self.dimension - len(self.name)) 
                for _ in texts]
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Mock single text embedding."""
        try:
            result = self.embed([text])
            return result[0] if result else None
        except Exception:
            return None
    
    def get_embedding_dimension(self) -> int:
        """Return mock dimension."""
        return self.dimension


class TestDualWriteEmbeddings:
    """Test dual-write embedding provider."""
    
    def test_dual_write_basic(self):
        """Test basic dual-write functionality."""
        primary = MockEmbeddingProvider("primary", dimension=4)
        shadow = MockEmbeddingProvider("shadow", dimension=4)
        
        dual_write = DualWriteEmbeddings(
            primary=primary,
            shadow=shadow,
            shadow_percentage=100.0  # Always send to shadow
        )
        
        texts = ["test1", "test2"]
        result = dual_write.embed(texts)
        
        # Should return primary results
        assert len(result) == 2
        assert result[0][:7] == [112.0, 114.0, 105.0, 109.0]  # "prim" in ASCII
        
        # Both providers should have been called
        assert len(primary.embed_calls) == 1
        assert len(shadow.embed_calls) == 1
        assert primary.embed_calls[0] == texts
        assert shadow.embed_calls[0] == texts
    
    def test_shadow_percentage(self):
        """Test shadow percentage functionality."""
        primary = MockEmbeddingProvider("primary")
        shadow = MockEmbeddingProvider("shadow")
        
        dual_write = DualWriteEmbeddings(
            primary=primary,
            shadow=shadow,
            shadow_percentage=0.0  # Never send to shadow
        )
        
        texts = ["test"]
        dual_write.embed(texts)
        
        # Only primary should be called
        assert len(primary.embed_calls) == 1
        assert len(shadow.embed_calls) == 0
    
    def test_primary_failure_propagation(self):
        """Test that primary failures are propagated."""
        primary = MockEmbeddingProvider("primary", fail_on_embed=True)
        shadow = MockEmbeddingProvider("shadow")
        
        dual_write = DualWriteEmbeddings(
            primary=primary,
            shadow=shadow,
            shadow_percentage=100.0
        )
        
        with pytest.raises(RuntimeError, match="primary embedding failed"):
            dual_write.embed(["test"])
        
        # Stats should reflect the failure
        stats = dual_write.get_stats()
        assert stats["primary_failures"] == 1
        assert stats["primary_requests"] == 1
    
    def test_shadow_failure_ignored(self):
        """Test that shadow failures don't affect primary results."""
        primary = MockEmbeddingProvider("primary", dimension=4)
        shadow = MockEmbeddingProvider("shadow", fail_on_embed=True)
        
        dual_write = DualWriteEmbeddings(
            primary=primary,
            shadow=shadow,
            shadow_percentage=100.0
        )
        
        texts = ["test"]
        result = dual_write.embed(texts)
        
        # Should still return primary results
        assert len(result) == 1
        assert result[0][:7] == [112.0, 114.0, 105.0, 109.0]  # "prim"
        
        # Stats should show shadow failure
        stats = dual_write.get_stats()
        assert stats["shadow_failures"] == 1
        assert stats["shadow_requests"] == 1
        assert stats["primary_failures"] == 0
    
    def test_statistics_tracking(self):
        """Test comprehensive statistics tracking."""
        primary = MockEmbeddingProvider("primary")
        shadow = MockEmbeddingProvider("shadow")
        
        dual_write = DualWriteEmbeddings(
            primary=primary,
            shadow=shadow,
            shadow_percentage=100.0
        )
        
        # Make some requests
        dual_write.embed(["test1"])
        dual_write.embed(["test2", "test3"])
        
        stats = dual_write.get_stats()
        
        # Check basic counts
        assert stats["primary_requests"] == 2
        assert stats["shadow_requests"] == 2
        assert stats["primary_failures"] == 0
        assert stats["shadow_failures"] == 0
        assert stats["comparisons"] == 2
        
        # Check success rates
        assert stats["primary_success_rate"] == 1.0
        assert stats["shadow_success_rate"] == 1.0
        
        # Check that timing info exists
        assert "primary_avg_time" in stats
        assert "shadow_avg_time" in stats
        assert stats["primary_avg_time"] >= 0
        assert stats["shadow_avg_time"] >= 0
    
    def test_embed_text_single(self):
        """Test single text embedding."""
        primary = MockEmbeddingProvider("primary", dimension=4)
        shadow = MockEmbeddingProvider("shadow", dimension=4)
        
        dual_write = DualWriteEmbeddings(primary=primary, shadow=shadow, shadow_percentage=100.0)
        
        result = dual_write.embed_text("test")
        
        assert result is not None
        assert len(result) == 4
        assert result[:7] == [112.0, 114.0, 105.0, 109.0]  # "prim"
    
    def test_get_embedding_dimension(self):
        """Test dimension retrieval from primary provider."""
        primary = MockEmbeddingProvider("primary", dimension=1024)
        shadow = MockEmbeddingProvider("shadow", dimension=512)
        
        dual_write = DualWriteEmbeddings(primary=primary, shadow=shadow)
        
        # Should return primary dimension
        assert dual_write.get_embedding_dimension() == 1024
    
    def test_no_shadow_provider(self):
        """Test dual-write with no shadow provider."""
        primary = MockEmbeddingProvider("primary")
        
        dual_write = DualWriteEmbeddings(
            primary=primary,
            shadow=None,
            shadow_percentage=100.0
        )
        
        result = dual_write.embed(["test"])
        
        # Should work normally with just primary
        assert len(result) == 1
        assert len(primary.embed_calls) == 1
        
        # Stats should show only primary
        stats = dual_write.get_stats()
        assert stats["primary_requests"] == 1
        assert stats["shadow_requests"] == 0
    
    def test_stats_reset(self):
        """Test statistics reset functionality."""
        primary = MockEmbeddingProvider("primary")
        shadow = MockEmbeddingProvider("shadow")
        
        dual_write = DualWriteEmbeddings(primary=primary, shadow=shadow, shadow_percentage=100.0)
        
        # Make some requests
        dual_write.embed(["test1"])
        dual_write.embed(["test2"])
        
        # Check stats are accumulated
        stats = dual_write.get_stats()
        assert stats["primary_requests"] == 2
        assert stats["shadow_requests"] == 2
        
        # Reset stats
        dual_write.reset_stats()
        
        # Check stats are reset
        stats = dual_write.get_stats()
        assert stats["primary_requests"] == 0
        assert stats["shadow_requests"] == 0
        assert stats["comparisons"] == 0
    
    def test_dynamic_shadow_percentage(self):
        """Test dynamic shadow percentage adjustment."""
        primary = MockEmbeddingProvider("primary")
        shadow = MockEmbeddingProvider("shadow")
        
        dual_write = DualWriteEmbeddings(
            primary=primary,
            shadow=shadow,
            shadow_percentage=0.0  # Start with no shadow
        )
        
        # Make request - should only hit primary
        dual_write.embed(["test1"])
        assert len(shadow.embed_calls) == 0
        
        # Change to 100% shadow
        dual_write.set_shadow_percentage(100.0)
        
        # Make another request - should hit both
        dual_write.embed(["test2"])
        assert len(shadow.embed_calls) == 1  # Now shadow is called
        assert len(primary.embed_calls) == 2  # Primary always called


class TestDualWriteFactory:
    """Test dual-write integration with factory."""
    
    @patch.dict(os.environ, {"EMBED_AB_WRITE": "voyage:50"})
    def test_factory_dualwrite_config(self):
        """Test factory dual-write configuration."""
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            # Mock providers
            mock_primary = MockEmbeddingProvider("openai")
            mock_shadow = MockEmbeddingProvider("voyage")
            # Fix lambda to accept both args (provider, config)
            mock_create.side_effect = lambda p, c=None: mock_primary if p == "openai" else mock_shadow
            
            with patch("integrations.providers.dualwrite.DualWriteEmbeddings") as mock_dual:
                mock_dual_instance = Mock()
                mock_dual.return_value = mock_dual_instance
                
                # Get embedder should create dual-write
                result = get_embedder(provider="openai")
                
                # Should have created DualWriteEmbeddings
                mock_dual.assert_called_once_with(
                    primary=mock_primary,
                    shadow=mock_shadow,
                    shadow_percentage=50.0,
                    compare_results=True
                )
                assert result == mock_dual_instance
    
    def test_create_dualwrite_provider_parsing(self):
        """Test dual-write provider configuration parsing."""
        primary = MockEmbeddingProvider("primary")
        
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            shadow = MockEmbeddingProvider("shadow")
            mock_create.return_value = shadow
            
            # Test "provider:percentage" format
            result = _create_dualwrite_provider(primary, "voyage:25")
            assert isinstance(result, DualWriteEmbeddings)
            assert result.shadow_percentage == 25.0
            
            # Test "provider" only format (should default to 100%)
            result = _create_dualwrite_provider(primary, "cohere")
            assert isinstance(result, DualWriteEmbeddings)
            assert result.shadow_percentage == 100.0
    
    def test_create_dualwrite_provider_shadow_unavailable(self):
        """Test dual-write when shadow provider is unavailable."""
        primary = MockEmbeddingProvider("primary")
        
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            # Shadow provider raises ImportError
            mock_create.side_effect = ImportError("voyageai not installed")
            
            # Should return primary provider only
            result = _create_dualwrite_provider(primary, "voyage:50")
            assert result == primary  # Fallback to primary only
    
    @patch.dict(os.environ, {"EMBED_AB_WRITE": "openai,voyage"})
    def test_get_embedder_ab_comma_format(self):
        """Test get_embedder_ab with 'primary,shadow' format."""
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            # Mock providers
            mock_openai = MockEmbeddingProvider("openai")
            mock_voyage = MockEmbeddingProvider("voyage")
            mock_create.side_effect = lambda p: mock_openai if p == "openai" else mock_voyage
            
            with patch("integrations.providers.dualwrite.DualWriteEmbeddings") as mock_dual:
                mock_dual_instance = Mock()
                mock_dual.return_value = mock_dual_instance
                
                # Get A/B embedder should create dual-write with both providers
                result = get_embedder_ab()
                
                # Should have created DualWriteEmbeddings with 100% shadow
                mock_dual.assert_called_once_with(
                    primary=mock_openai,
                    shadow=mock_voyage, 
                    shadow_percentage=100.0,
                    compare_results=True
                )
                assert result == mock_dual_instance
    
    @patch.dict(os.environ, {"EMBED_AB_WRITE": "voyage:25"})
    def test_get_embedder_ab_percentage_format(self):
        """Test get_embedder_ab with 'shadow:percentage' format."""
        with patch("integrations.embeddings_factory._create_provider") as mock_create:
            mock_openai = MockEmbeddingProvider("openai")
            mock_voyage = MockEmbeddingProvider("voyage")
            mock_create.side_effect = lambda p: mock_openai if p == "openai" else mock_voyage
            
            with patch("integrations.providers.dualwrite.DualWriteEmbeddings") as mock_dual:
                mock_dual_instance = Mock()
                mock_dual.return_value = mock_dual_instance
                
                result = get_embedder_ab()
                
                # Should use openai as primary (from default) and voyage as shadow with 25%
                mock_dual.assert_called_once_with(
                    primary=mock_openai,
                    shadow=mock_voyage,
                    shadow_percentage=25.0,
                    compare_results=True
                )
                assert result == mock_dual_instance
    
    @patch.dict(os.environ, {"EMBED_AB_WRITE": ""})
    def test_get_embedder_ab_no_config(self):
        """Test get_embedder_ab returns standard embedder when no A/B config."""
        with patch("integrations.embeddings_factory.get_embedder") as mock_get:
            mock_embedder = Mock()
            mock_get.return_value = mock_embedder
            
            result = get_embedder_ab()
            
            # Should fall back to standard factory
            mock_get.assert_called_once()
            assert result == mock_embedder


class TestDualWriteComparison:
    """Test dual-write embedding comparison functionality."""
    
    def test_comparison_with_numpy(self):
        """Test embedding comparison with numpy available."""
        try:
            import numpy as np
            numpy_available = True
        except ImportError:
            numpy_available = False
            
        if not numpy_available:
            pytest.skip("numpy not available")
        
        # Create providers with different but predictable results
        primary = MockEmbeddingProvider("prim", dimension=4)  # ASCII: [112, 114, 105, 109]
        shadow = MockEmbeddingProvider("shad", dimension=4)   # ASCII: [115, 104, 97, 100]
        
        dual_write = DualWriteEmbeddings(
            primary=primary,
            shadow=shadow,
            shadow_percentage=100.0,
            compare_results=True
        )
        
        # Make request to generate comparison
        dual_write.embed(["test"])
        
        stats = dual_write.get_stats()
        
        # Should have comparison data
        assert stats["comparisons"] == 1
        assert len(stats["differences"]) == 1
        
        # Check comparison results
        comparison = stats["differences"][0]
        assert "differences" in comparison
        assert len(comparison["differences"]) == 1
        
        diff = comparison["differences"][0]
        assert "cosine_similarity" in diff
        assert "l2_distance" in diff
        assert isinstance(diff["cosine_similarity"], float)
        assert isinstance(diff["l2_distance"], float)
    
    def test_comparison_without_numpy(self):
        """Test embedding comparison fallback without numpy."""
        primary = MockEmbeddingProvider("primary")
        shadow = MockEmbeddingProvider("shadow")
        
        dual_write = DualWriteEmbeddings(
            primary=primary,
            shadow=shadow,
            shadow_percentage=100.0,
            compare_results=True
        )
        
        # Mock numpy import failure by patching the comparison method
        with patch.object(dual_write, '_compare_embeddings') as mock_compare:
            mock_compare.return_value = {"message": "numpy not available for detailed comparison"}
            dual_write.embed(["test"])
        
        stats = dual_write.get_stats()
        
        # Should have attempted comparison
        assert stats["comparisons"] == 1
        
        # But comparison should contain fallback message
        comparison = stats["differences"][0]
        assert "message" in comparison or "error" in comparison
    
    def test_empty_results_comparison(self):
        """Test comparison handling with empty results."""
        primary = Mock()
        shadow = Mock()
        
        dual_write = DualWriteEmbeddings(primary=primary, shadow=shadow)
        
        # Test comparison with empty results
        comparison = dual_write._compare_embeddings([], [])
        assert "error" in comparison
        
        comparison = dual_write._compare_embeddings([[1, 2]], [])
        assert "error" in comparison
    
    def test_dimension_mismatch_comparison(self):
        """Test comparison with dimension mismatches."""
        primary = Mock()
        shadow = Mock()
        
        dual_write = DualWriteEmbeddings(primary=primary, shadow=shadow)
        
        # Test with different vector lengths
        primary_result = [[1, 2, 3]]
        shadow_result = [[1, 2, 3, 4]]
        
        comparison = dual_write._compare_embeddings(primary_result, shadow_result)
        
        assert "differences" in comparison
        diff_msg = comparison["differences"][0]
        assert "dimension mismatch" in diff_msg


if __name__ == "__main__":
    pytest.main([__file__])