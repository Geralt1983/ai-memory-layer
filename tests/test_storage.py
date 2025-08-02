import pytest
import numpy as np
import tempfile
import os
from core.memory_engine import Memory
from storage.faiss_store import FaissVectorStore
from storage.chroma_store import ChromaVectorStore


class TestFaissVectorStore:
    """Test cases for FAISS vector store"""
    
    def test_faiss_store_creation(self):
        """Test basic FAISS store creation"""
        store = FaissVectorStore(dimension=128)
        
        assert store.dimension == 128
        assert store.index.ntotal == 0
        assert store.current_id == 0
        assert len(store.memories) == 0
    
    def test_add_memory_to_faiss(self, faiss_store):
        """Test adding memory to FAISS store"""
        memory = Memory(
            content="Test memory",
            embedding=np.random.rand(10).astype('float32')
        )
        
        memory_id = faiss_store.add_memory(memory)
        
        assert faiss_store.index.ntotal == 1
        assert memory_id == "0"
        assert int(memory_id) in faiss_store.memories
    
    def test_add_memory_without_embedding(self, faiss_store):
        """Test that adding memory without embedding raises error"""
        memory = Memory(content="Test memory")  # No embedding
        
        with pytest.raises(ValueError, match="Memory must have an embedding"):
            faiss_store.add_memory(memory)
    
    def test_search_faiss_empty(self, faiss_store):
        """Test search on empty FAISS store"""
        query_embedding = np.random.rand(10).astype('float32')
        results = faiss_store.search(query_embedding, k=5)
        
        assert results == []
    
    def test_search_faiss_with_memories(self, faiss_store):
        """Test search with memories in FAISS store"""
        # Add some memories
        for i in range(3):
            memory = Memory(
                content=f"Memory {i}",
                embedding=np.random.rand(10).astype('float32')
            )
            faiss_store.add_memory(memory)
        
        query_embedding = np.random.rand(10).astype('float32')
        results = faiss_store.search(query_embedding, k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, Memory) for r in results)
        assert all(hasattr(r, 'relevance_score') for r in results)
        assert all(r.relevance_score > 0 for r in results)
    
    def test_delete_memory_faiss(self, faiss_store):
        """Test deleting memory from FAISS store"""
        memory = Memory(
            content="Test memory",
            embedding=np.random.rand(10).astype('float32')
        )
        
        memory_id = faiss_store.add_memory(memory)
        assert len(faiss_store.memories) == 1
        
        success = faiss_store.delete_memory(memory_id)
        assert success
        assert len(faiss_store.memories) == 0
    
    def test_delete_nonexistent_memory_faiss(self, faiss_store):
        """Test deleting non-existent memory"""
        success = faiss_store.delete_memory("999")
        assert not success
    
    def test_faiss_persistence(self, temp_dir):
        """Test FAISS persistence functionality"""
        index_path = os.path.join(temp_dir, "test_index")
        
        # Create store and add memory
        store1 = FaissVectorStore(dimension=10, index_path=index_path)
        memory = Memory(
            content="Persistent memory",
            embedding=np.random.rand(10).astype('float32')
        )
        store1.add_memory(memory)
        
        # Create new store that should load the index
        store2 = FaissVectorStore(dimension=10, index_path=index_path)
        
        assert store2.index.ntotal == 1
        assert len(store2.memories) == 1
        assert store2.memories[0].content == "Persistent memory"


class TestChromaVectorStore:
    """Test cases for ChromaDB vector store"""
    
    def test_chroma_store_creation(self, temp_dir):
        """Test basic ChromaDB store creation"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        assert store.collection_name == "test_collection"
        assert store.persist_directory == temp_dir
        assert store.collection is not None
    
    def test_add_memory_to_chroma(self, temp_dir):
        """Test adding memory to ChromaDB store"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        memory = Memory(
            content="Test memory",
            embedding=np.random.rand(384).astype('float32'),  # ChromaDB default dimension
            metadata={"type": "test"}
        )
        
        memory_id = store.add_memory(memory)
        
        assert memory_id is not None
        assert isinstance(memory_id, str)
    
    def test_add_memory_chroma_without_embedding(self, temp_dir):
        """Test adding memory without embedding to ChromaDB"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        memory = Memory(content="Test memory")  # No embedding
        
        with pytest.raises(ValueError, match="Memory must have an embedding"):
            store.add_memory(memory)
    
    def test_search_chroma_empty(self, temp_dir):
        """Test search on empty ChromaDB store"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        query_embedding = np.random.rand(384).astype('float32')
        results = store.search(query_embedding, k=5)
        
        assert results == []
    
    def test_search_chroma_with_memories(self, temp_dir):
        """Test search with memories in ChromaDB store"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        # Add some memories
        for i in range(3):
            memory = Memory(
                content=f"Memory {i}",
                embedding=np.random.rand(384).astype('float32'),
                metadata={"index": i}
            )
            store.add_memory(memory)
        
        query_embedding = np.random.rand(384).astype('float32')
        results = store.search(query_embedding, k=2)
        
        assert len(results) <= 2  # Might be less if collection is small
        assert all(isinstance(r, Memory) for r in results)
    
    def test_delete_memory_chroma(self, temp_dir):
        """Test deleting memory from ChromaDB store"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        memory = Memory(
            content="Test memory",
            embedding=np.random.rand(384).astype('float32')
        )
        
        memory_id = store.add_memory(memory)
        success = store.delete_memory(memory_id)
        
        assert success
    
    def test_delete_nonexistent_memory_chroma(self, temp_dir):
        """Test deleting non-existent memory from ChromaDB"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        success = store.delete_memory("nonexistent_id")
        assert not success