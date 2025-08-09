"""
Test FAISS dimension validation and sidecar metadata system
"""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import patch

from storage.faiss_store import FaissVectorStore, IndexMeta
from core.memory_engine import Memory


class TestFAISSDimensionGuard:
    """Test FAISS dimension validation and metadata persistence"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "test_index")
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_new_index_creation(self):
        """Test creating a new index creates proper metadata files"""
        store = FaissVectorStore(self.index_path)
        
        # Check that metadata files exist
        assert os.path.exists(f"{self.index_path}.meta.json")
        assert os.path.exists(f"{self.index_path}.ids.json")
        
        # Check default dimension
        meta = store._load_meta()
        assert meta.dim == 1536
        assert meta.next_id == 0
    
    def test_dimension_validation_success(self):
        """Test that matching dimensions work correctly"""
        store = FaissVectorStore(self.index_path)
        
        # Create memory with correct dimension
        embedding = np.random.rand(1536).tolist()
        memory = Memory(content="test", embedding=embedding)
        
        # Should succeed
        memory_id = store.add_memory(memory)
        assert memory_id is not None
        assert store.current_id == 1
    
    def test_dimension_validation_failure(self):
        """Test that dimension mismatch raises ValueError"""
        store = FaissVectorStore(self.index_path)
        
        # Add a memory with correct dimension first
        correct_embedding = np.random.rand(1536).tolist()
        correct_memory = Memory(content="correct", embedding=correct_embedding)
        store.add_memory(correct_memory)
        
        # Now try to add memory with wrong dimension
        wrong_embedding = np.random.rand(512).tolist()  # Wrong dimension
        wrong_memory = Memory(content="wrong", embedding=wrong_embedding)
        
        with pytest.raises(ValueError, match="Embedding dimension changed"):
            store.add_memory(wrong_memory)
    
    def test_empty_index_dimension_change(self):
        """Test that dimension can change on empty index"""
        store = FaissVectorStore(self.index_path)
        
        # Add memory with different dimension to empty index
        new_embedding = np.random.rand(768).tolist()  # Different dimension
        memory = Memory(content="test", embedding=new_embedding)
        
        # Should succeed and update dimension
        memory_id = store.add_memory(memory)
        assert memory_id is not None
        
        # Check that dimension was updated
        meta = store._load_meta()
        assert meta.dim == 768
    
    def test_metadata_persistence(self):
        """Test that metadata is properly persisted and loaded"""
        # Create store and add memory
        store = FaissVectorStore(self.index_path)
        embedding = np.random.rand(1536).tolist()
        memory = Memory(content="test", embedding=embedding)
        store.add_memory(memory)
        
        # Create new store instance from same path
        new_store = FaissVectorStore(self.index_path)
        
        # Check that metadata was loaded correctly
        assert new_store.current_id == 1
        meta = new_store._load_meta()
        assert meta.dim == 1536
        assert meta.next_id == 1
    
    def test_id_mapping_persistence(self):
        """Test that ID mapping is properly persisted"""
        store = FaissVectorStore(self.index_path)
        
        # Add multiple memories
        memories = []
        for i in range(3):
            embedding = np.random.rand(1536).tolist()
            memory = Memory(content=f"test_{i}", embedding=embedding)
            memory.id = f"mem_{i}"
            memory_id = store.add_memory(memory)
            memories.append((memory_id, memory.id))
        
        # Create new store and check ID mapping
        new_store = FaissVectorStore(self.index_path)
        
        for memory_id, original_id in memories:
            assert memory_id in new_store.id_to_mem
            assert new_store.id_to_mem[memory_id] == original_id
    
    @patch('storage.faiss_store.FAISS_AVAILABLE', False)
    def test_compatibility_mode(self):
        """Test that store works in compatibility mode without FAISS"""
        store = FaissVectorStore(self.index_path)
        
        # Should still create metadata files
        assert os.path.exists(f"{self.index_path}.meta.json")
        
        # Adding memory should work without errors
        memory = Memory(content="test", embedding=[0.1] * 1536)
        memory_id = store.add_memory(memory)
        assert memory_id is not None
    
    def test_search_with_new_format(self):
        """Test search functionality with new sidecar format"""
        store = FaissVectorStore(self.index_path)
        
        # Add test memories
        embeddings = [np.random.rand(1536).tolist() for _ in range(5)]
        memories = []
        
        for i, embedding in enumerate(embeddings):
            memory = Memory(content=f"test content {i}", embedding=embedding)
            store.add_memory(memory)
            memories.append(memory)
        
        # Search with query
        query_embedding = np.random.rand(1536).tolist()
        results = store.search(query_embedding, k=3)
        
        # Should get results
        assert len(results) <= 3
        assert len(results) > 0
        
        # Results should have relevance scores
        for result in results:
            assert hasattr(result, 'relevance_score')
            assert isinstance(result.relevance_score, float)
    
    def test_legacy_compatibility(self):
        """Test that new format can load legacy .pkl files"""
        # First create a legacy-style store manually
        import pickle
        
        legacy_memories = {
            0: Memory(content="legacy test", embedding=np.random.rand(1536).tolist()),
            1: Memory(content="another legacy", embedding=np.random.rand(1536).tolist())
        }
        
        # Save in legacy format
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump({
                "memories": legacy_memories,
                "current_id": 2,
                "dimension": 1536
            }, f)
        
        # Create metadata files for the new format
        meta = IndexMeta(dim=1536, next_id=2)
        import json
        with open(f"{self.index_path}.meta.json", "w") as f:
            json.dump({"dim": meta.dim, "next_id": meta.next_id}, f)
        
        with open(f"{self.index_path}.ids.json", "w") as f:
            json.dump({}, f)
        
        # Create a mock index file
        if os.getenv('SKIP_FAISS_FILE_TEST') != 'true':
            # Only create if FAISS is available
            try:
                import faiss
                index = faiss.IndexFlatIP(1536)
                faiss.write_index(index, f"{self.index_path}.index")
            except ImportError:
                pass
        
        # Load with new store
        store = FaissVectorStore(self.index_path)
        
        # Should load legacy memories
        assert len(store.memories) == 2
        assert store.current_id == 2
        
        # Check that we can still add new memories
        new_memory = Memory(content="new memory", embedding=np.random.rand(1536).tolist())
        memory_id = store.add_memory(new_memory)
        assert memory_id == "2"
        assert store.current_id == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])