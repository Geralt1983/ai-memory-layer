"""Tests for FAISS index implementation."""

import pytest
import tempfile
import os
import json
import numpy as np
from pathlib import Path

from memory_layer.index.faiss_index import FAISSIndex, IndexSpec


@pytest.mark.unit
class TestIndexSpec:
    """Test IndexSpec dataclass."""
    
    def test_index_spec_creation(self):
        """Test IndexSpec creation and defaults."""
        spec = IndexSpec(
            provider="openai",
            model="text-embedding-3-small", 
            dim=1536,
            normalize=True
        )
        
        assert spec.provider == "openai"
        assert spec.model == "text-embedding-3-small"
        assert spec.dim == 1536
        assert spec.normalize is True
        assert spec.metric == "IP"  # Default value
    
    def test_index_spec_custom_metric(self):
        """Test IndexSpec with custom metric."""
        spec = IndexSpec(
            provider="voyage",
            model="voyage-2",
            dim=1024,
            normalize=False,
            metric="L2"
        )
        
        assert spec.metric == "L2"
    
    def test_index_spec_frozen(self):
        """Test that IndexSpec is frozen/immutable."""
        spec = IndexSpec(provider="test", model="test", dim=512, normalize=True)
        
        with pytest.raises(Exception):  # Should be frozen dataclass
            spec.provider = "changed"


@pytest.mark.unit
class TestFAISSIndex:
    """Test FAISS index implementation."""
    
    def test_metric_selection(self):
        """Test that correct FAISS metric is selected."""
        import faiss
        
        # Test IP metric
        spec_ip = IndexSpec(provider="test", model="test", dim=2, normalize=True, metric="IP")
        index_ip = FAISSIndex("test.index", spec_ip)
        assert index_ip._metric() == faiss.METRIC_INNER_PRODUCT
        
        # Test L2 metric
        spec_l2 = IndexSpec(provider="test", model="test", dim=2, normalize=False, metric="L2")
        index_l2 = FAISSIndex("test.index", spec_l2)
        assert index_l2._metric() == faiss.METRIC_L2
    
    def test_digest_computation(self):
        """Test that digest includes both vectors and spec."""
        spec = IndexSpec(provider="test", model="test", dim=2, normalize=True)
        index = FAISSIndex("test.index", spec)
        
        vectors1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        vectors2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32") 
        different_vectors = np.array([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
        
        digest1 = index._digest(vectors1)
        digest2 = index._digest(vectors2)
        digest3 = index._digest(different_vectors)
        
        # Same vectors should produce same digest
        assert digest1 == digest2
        # Different vectors should produce different digest
        assert digest1 != digest3
    
    def test_digest_includes_spec(self):
        """Test that digest changes when spec changes."""
        vectors = np.array([[1.0, 2.0]], dtype="float32")
        
        spec1 = IndexSpec(provider="test", model="model1", dim=2, normalize=True)
        spec2 = IndexSpec(provider="test", model="model2", dim=2, normalize=True)
        
        index1 = FAISSIndex("test.index", spec1)
        index2 = FAISSIndex("test.index", spec2)
        
        digest1 = index1._digest(vectors)
        digest2 = index2._digest(vectors)
        
        # Different specs should produce different digests
        assert digest1 != digest2
    
    def test_load_or_build_creates_index(self):
        """Test that load_or_build creates new index when none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            spec = IndexSpec(provider="test", model="test", dim=2, normalize=False, metric="L2")
            
            index = FAISSIndex(index_path, spec)
            vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
            ids = ["id1", "id2"]
            
            faiss_index = index.load_or_build(vectors, ids)
            
            # Should create index file
            assert os.path.exists(index_path)
            
            # Should create metadata file
            meta_path = os.path.join(tmpdir, "faiss.index.meta.json")
            assert os.path.exists(meta_path)
            
            # Check metadata content
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            assert meta["count"] == 2
            assert "digest" in meta
            assert meta["spec"]["provider"] == "test"
    
    def test_load_or_build_reuses_existing(self):
        """Test that load_or_build reuses existing index when digest matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            spec = IndexSpec(provider="test", model="test", dim=2, normalize=False, metric="L2")
            
            index = FAISSIndex(index_path, spec)
            vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
            ids = ["id1", "id2"]
            
            # First build
            faiss_index1 = index.load_or_build(vectors, ids)
            original_mtime = os.path.getmtime(index_path)
            
            # Small delay to ensure different mtime if rebuilt
            import time
            time.sleep(0.1)
            
            # Second build with same vectors should reuse
            faiss_index2 = index.load_or_build(vectors, ids)
            new_mtime = os.path.getmtime(index_path)
            
            # File should not have been rebuilt
            assert original_mtime == new_mtime
    
    def test_load_or_build_rebuilds_on_different_vectors(self):
        """Test that index is rebuilt when vectors change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index") 
            spec = IndexSpec(provider="test", model="test", dim=2, normalize=False, metric="L2")
            
            index = FAISSIndex(index_path, spec)
            
            # First build
            vectors1 = np.array([[1.0, 2.0]], dtype="float32")
            ids1 = ["id1"]
            index.load_or_build(vectors1, ids1)
            original_mtime = os.path.getmtime(index_path)
            
            import time
            time.sleep(0.1)
            
            # Second build with different vectors
            vectors2 = np.array([[3.0, 4.0]], dtype="float32") 
            ids2 = ["id2"]
            index.load_or_build(vectors2, ids2)
            new_mtime = os.path.getmtime(index_path)
            
            # File should have been rebuilt
            assert new_mtime > original_mtime
    
    def test_search_static_method(self):
        """Test the static search method."""
        import faiss
        
        # Create simple index
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32")
        index = faiss.IndexFlatL2(2)
        index.add(vectors)
        
        # Search for vector similar to first one
        query = np.array([[0.9, 0.1]], dtype="float32")
        D, I = FAISSIndex.search(index, query, k=2)
        
        # Should find closest matches
        assert I[0][0] == 0  # First vector should be closest
        assert len(I[0]) == 2  # Should return k=2 results
    
    def test_directory_creation(self):
        """Test that parent directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nested path that doesn't exist
            nested_path = os.path.join(tmpdir, "nested", "subdir", "test.index")
            spec = IndexSpec(provider="test", model="test", dim=2, normalize=False)
            
            index = FAISSIndex(nested_path, spec)
            vectors = np.array([[1.0, 2.0]], dtype="float32")
            ids = ["id1"]
            
            # Should create nested directories
            index.load_or_build(vectors, ids)
            
            assert os.path.exists(nested_path)
            assert os.path.exists(os.path.dirname(nested_path))


@pytest.mark.integration 
class TestFAISSIndexIntegration:
    """Integration tests for FAISS index."""
    
    def test_end_to_end_index_workflow(self):
        """Test complete index creation and search workflow."""
        import faiss
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "integration.index")
            spec = IndexSpec(
                provider="test", 
                model="test-model",
                dim=3,
                normalize=True,
                metric="IP"
            )
            
            # Create test data
            vectors = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], 
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0]
            ], dtype="float32")
            
            # Normalize since we're using IP metric
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            
            ids = ["vec1", "vec2", "vec3", "vec4"]
            
            # Build index
            index = FAISSIndex(index_path, spec)
            faiss_index = index.load_or_build(vectors, ids)
            
            # Test search
            query = np.array([[0.9, 0.1, 0.0]], dtype="float32")
            query = query / np.linalg.norm(query)
            
            D, I = FAISSIndex.search(faiss_index, query, k=2)
            
            # First result should be vec1 (closest to [1,0,0])
            assert I[0][0] == 0
            assert len(D[0]) == 2
            assert len(I[0]) == 2
    
    def test_persistence_across_instances(self):
        """Test that index persists across different FAISSIndex instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "persist.index")
            spec = IndexSpec(provider="test", model="test", dim=2, normalize=False)
            
            vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
            ids = ["id1", "id2"]
            
            # Create and build index with first instance
            index1 = FAISSIndex(index_path, spec)
            index1.load_or_build(vectors, ids)
            
            # Create new instance and verify it loads existing index
            index2 = FAISSIndex(index_path, spec)
            faiss_index = index2.load_or_build(vectors, ids)
            
            # Should have loaded existing index (2 vectors)
            assert faiss_index.ntotal == 2
    
    def test_spec_change_forces_rebuild(self):
        """Test that changing spec forces index rebuild."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "rebuild.index")
            
            vectors = np.array([[1.0, 2.0]], dtype="float32")
            ids = ["id1"]
            
            # Build with first spec
            spec1 = IndexSpec(provider="test", model="model1", dim=2, normalize=False)
            index1 = FAISSIndex(index_path, spec1)
            index1.load_or_build(vectors, ids)
            original_mtime = os.path.getmtime(index_path)
            
            import time
            time.sleep(0.1)
            
            # Build with different spec (should force rebuild)
            spec2 = IndexSpec(provider="test", model="model2", dim=2, normalize=False)
            index2 = FAISSIndex(index_path, spec2)
            index2.load_or_build(vectors, ids)
            new_mtime = os.path.getmtime(index_path)
            
            # Should have been rebuilt
            assert new_mtime > original_mtime