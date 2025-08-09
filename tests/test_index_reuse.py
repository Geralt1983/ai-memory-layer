"""Tests for FAISS index reuse and caching functionality."""

import pytest
import tempfile
import os
import json
import numpy as np
from memory_layer.index.faiss_index import FAISSIndex, IndexSpec, _meta_digest


@pytest.mark.unit
class TestFAISSIndexReuse:
    """Test FAISS index metadata and reuse functionality."""
    
    def test_index_spec_creation(self):
        """Test IndexSpec dataclass creation and properties."""
        spec = IndexSpec(
            provider="voyage",
            model="voyage-3",
            dim=1024,
            normalize=True,
            metric="IP"
        )
        
        assert spec.provider == "voyage"
        assert spec.model == "voyage-3"
        assert spec.dim == 1024
        assert spec.normalize is True
        assert spec.metric == "IP"
    
    def test_meta_digest_consistency(self):
        """Test that meta digest is consistent for same inputs."""
        spec = IndexSpec("voyage", "voyage-3", 1024, True)
        corpus_hashes = ["hash1", "hash2", "hash3"]
        
        digest1 = _meta_digest(spec, corpus_hashes)
        digest2 = _meta_digest(spec, corpus_hashes)
        
        assert digest1 == digest2
        assert isinstance(digest1, str)
        assert len(digest1) == 64  # SHA256 hex length
    
    def test_meta_digest_changes_with_spec(self):
        """Test that meta digest changes when spec changes."""
        spec1 = IndexSpec("voyage", "voyage-3", 1024, True)
        spec2 = IndexSpec("voyage", "voyage-3", 1024, False)  # normalize changed
        corpus_hashes = ["hash1", "hash2"]
        
        digest1 = _meta_digest(spec1, corpus_hashes)
        digest2 = _meta_digest(spec2, corpus_hashes)
        
        assert digest1 != digest2
    
    def test_meta_digest_changes_with_corpus(self):
        """Test that meta digest changes when corpus changes."""
        spec = IndexSpec("voyage", "voyage-3", 1024, True)
        corpus1 = ["hash1", "hash2"]
        corpus2 = ["hash1", "hash3"]  # Different hash
        
        digest1 = _meta_digest(spec, corpus1)
        digest2 = _meta_digest(spec, corpus2)
        
        assert digest1 != digest2
    
    def test_index_creation_and_metadata(self):
        """Test index creation with metadata tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            spec = IndexSpec("test", "model", 3, True, "IP")
            faiss_index = FAISSIndex(index_path, spec)
            
            # Create test vectors
            vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            ids = ["id1", "id2"]
            
            # Build index
            index = faiss_index.load_or_build(vectors, ids)
            
            # Check index properties
            assert index.ntotal == 2
            assert index.d == 3
            
            # Check metadata file exists
            meta_path = os.path.join(tmpdir, "faiss.index.meta.json")
            assert os.path.exists(meta_path)
            
            # Check metadata content
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            assert "digest" in meta
            assert meta["count"] == 2
            assert meta["spec"]["provider"] == "test"
            assert meta["spec"]["model"] == "model"
    
    def test_index_reuse_same_digest(self):
        """Test that index is reused when digest matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            spec = IndexSpec("test", "model", 3, True, "IP")
            faiss_index = FAISSIndex(index_path, spec)
            
            # Create test vectors
            vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            ids = ["id1", "id2"]
            
            # First build
            index1 = faiss_index.load_or_build(vectors, ids)
            original_index_mtime = os.path.getmtime(index_path)
            
            # Second call with same data should reuse
            import time
            time.sleep(0.1)  # Ensure different mtime if rebuilt
            index2 = faiss_index.load_or_build(vectors, ids)
            
            # Index file should not have been modified
            assert os.path.getmtime(index_path) == original_index_mtime
            
            # Both indexes should have same properties
            assert index1.ntotal == index2.ntotal == 2
    
    def test_index_rebuild_different_digest(self):
        """Test that index is rebuilt when digest changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            spec = IndexSpec("test", "model", 3, True, "IP")
            faiss_index = FAISSIndex(index_path, spec)
            
            # First build with 2 vectors
            vectors1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            ids1 = ["id1", "id2"]
            index1 = faiss_index.load_or_build(vectors1, ids1)
            assert index1.ntotal == 2
            
            original_index_mtime = os.path.getmtime(index_path)
            
            # Second build with different vectors (should rebuild)
            import time
            time.sleep(0.1)
            vectors2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            ids2 = ["id1", "id2", "id3"]
            index2 = faiss_index.load_or_build(vectors2, ids2)
            
            # Index should have been rebuilt
            assert index2.ntotal == 3
            assert os.path.getmtime(index_path) > original_index_mtime
    
    def test_index_search_functionality(self):
        """Test index search functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            spec = IndexSpec("test", "model", 3, True, "IP")
            faiss_index = FAISSIndex(index_path, spec)
            
            # Create normalized test vectors for inner product
            vectors = np.array([
                [1.0, 0.0, 0.0],  # Unit vector along x
                [0.0, 1.0, 0.0],  # Unit vector along y  
                [0.0, 0.0, 1.0],  # Unit vector along z
            ], dtype=np.float32)
            ids = ["x", "y", "z"]
            
            # Build index
            index = faiss_index.load_or_build(vectors, ids)
            
            # Search for vector similar to first one
            query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            distances, indices = FAISSIndex.search(index, query, k=2)
            
            # First result should be the exact match (index 0)
            assert indices[0][0] == 0
            assert distances[0][0] > 0.99  # Should be very close to 1.0 for inner product
    
    def test_different_metrics(self):
        """Test index creation with different distance metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test Inner Product (IP) metric
            spec_ip = IndexSpec("test", "model", 3, True, "IP")
            faiss_index_ip = FAISSIndex(os.path.join(tmpdir, "ip.index"), spec_ip)
            
            # Test L2 metric
            spec_l2 = IndexSpec("test", "model", 3, False, "L2")
            faiss_index_l2 = FAISSIndex(os.path.join(tmpdir, "l2.index"), spec_l2)
            
            vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            ids = ["id1", "id2"]
            
            # Both should build successfully
            index_ip = faiss_index_ip.load_or_build(vectors, ids)
            index_l2 = faiss_index_l2.load_or_build(vectors, ids)
            
            assert index_ip.ntotal == 2
            assert index_l2.ntotal == 2
    
    def test_corrupted_metadata_handling(self):
        """Test handling of corrupted metadata files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            meta_path = os.path.join(tmpdir, "faiss.index.meta.json")
            spec = IndexSpec("test", "model", 3, True, "IP")
            faiss_index = FAISSIndex(index_path, spec)
            
            vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            ids = ["id1"]
            
            # Create corrupted metadata file
            with open(meta_path, 'w') as f:
                f.write("invalid json content")
            
            # Should still build successfully (ignores corrupted meta)
            index = faiss_index.load_or_build(vectors, ids)
            assert index.ntotal == 1
            
            # Should create new valid metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            assert "digest" in meta
    
    def test_missing_index_file_with_metadata(self):
        """Test handling when metadata exists but index file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test.index")
            meta_path = os.path.join(tmpdir, "faiss.index.meta.json")
            spec = IndexSpec("test", "model", 3, True, "IP")
            faiss_index = FAISSIndex(index_path, spec)
            
            vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            ids = ["id1"]
            
            # Create metadata file without index file
            meta_data = {"digest": "fake-digest", "count": 1, "spec": spec.__dict__}
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f)
            
            # Should rebuild since index file is missing
            index = faiss_index.load_or_build(vectors, ids)
            assert index.ntotal == 1
            assert os.path.exists(index_path)


@pytest.mark.integration
class TestFAISSIndexIntegration:
    """Integration tests for FAISS index with real-world scenarios."""
    
    def test_large_index_reuse(self):
        """Test index reuse with larger dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "large.index")
            spec = IndexSpec("test", "model", 128, True, "IP")
            faiss_index = FAISSIndex(index_path, spec)
            
            # Create larger dataset
            np.random.seed(42)  # For reproducible tests
            vectors = np.random.rand(1000, 128).astype(np.float32)
            # Normalize for inner product
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
            ids = [f"doc_{i}" for i in range(1000)]
            
            # First build
            import time
            start_time = time.time()
            index1 = faiss_index.load_or_build(vectors, ids)
            first_build_time = time.time() - start_time
            
            assert index1.ntotal == 1000
            
            # Second build (should be much faster due to reuse)
            start_time = time.time()
            index2 = faiss_index.load_or_build(vectors, ids)
            second_build_time = time.time() - start_time
            
            assert index2.ntotal == 1000
            # Second build should be significantly faster (at least 5x)
            assert second_build_time < first_build_time / 5
    
    def test_concurrent_access_safety(self):
        """Test that concurrent access doesn't corrupt the index."""
        import threading
        import queue
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "concurrent.index")
            spec = IndexSpec("test", "model", 3, True, "IP")
            
            vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            ids = ["id1", "id2"]
            
            results = queue.Queue()
            errors = queue.Queue()
            
            def build_index():
                try:
                    faiss_index = FAISSIndex(index_path, spec)
                    index = faiss_index.load_or_build(vectors, ids)
                    results.put(index.ntotal)
                except Exception as e:
                    errors.put(e)
            
            # Start multiple threads
            threads = [threading.Thread(target=build_index) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # All should succeed
            assert errors.empty(), f"Errors occurred: {list(errors.queue)}"
            
            # All should return same result
            result_counts = list(results.queue)
            assert all(count == 2 for count in result_counts)