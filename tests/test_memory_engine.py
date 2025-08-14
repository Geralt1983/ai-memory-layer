import pytest
import json
import os
import threading
from unittest.mock import Mock, patch
from core.memory_engine import MemoryEngine, Memory
from storage.faiss_store import FaissVectorStore


class TestMemoryEngine:
    """Test cases for the MemoryEngine class"""

    def test_memory_engine_creation(self):
        """Test basic memory engine creation"""
        engine = MemoryEngine()

        assert engine.vector_store is None
        assert engine.embedding_provider is None
        assert engine.persist_path is None
        assert engine.memories == []

    def test_memory_engine_with_dependencies(
        self, faiss_store, mock_embedding_provider, temp_file
    ):
        """Test memory engine creation with all dependencies"""
        engine = MemoryEngine(
            vector_store=faiss_store,
            embedding_provider=mock_embedding_provider,
            persist_path=temp_file,
        )

        assert engine.vector_store == faiss_store
        assert engine.embedding_provider == mock_embedding_provider
        assert engine.persist_path == temp_file

    def test_add_memory_basic(self):
        """Test adding a memory without dependencies"""
        engine = MemoryEngine()
        memory = engine.add_memory("Test content", {"type": "test"})

        assert len(engine.memories) == 1
        assert memory.content == "Test content"
        assert memory.metadata == {"type": "test"}
        assert memory in engine.memories

    def test_add_memory_with_embedding(self, memory_engine):
        """Test adding memory with embedding generation"""
        memory = memory_engine.add_memory("Test content")

        assert len(memory_engine.memories) == 1
        assert memory.embedding is not None
        assert len(memory.embedding) == 10  # Mock embedding dimension

    def test_add_memory_with_vector_store(self, memory_engine):
        """Test that memory is added to vector store"""
        memory = memory_engine.add_memory("Test content")

        # Check that vector store received the memory
        assert memory_engine.vector_store.index.ntotal == 1

    def test_search_memories_without_dependencies(self):
        """Test search fallback when no vector store or embedding provider"""
        engine = MemoryEngine()
        engine.add_memory("First memory")
        engine.add_memory("Second memory")

        results = engine.search_memories("test query", k=1)

        assert len(results) == 1
        assert results[0].content == "Second memory"  # Most recent

    def test_search_memories_with_dependencies(self, memory_engine):
        """Test semantic search with vector store and embedding provider"""
        # Add some test memories
        memory_engine.add_memory("Python programming tutorial")
        memory_engine.add_memory("JavaScript web development")
        memory_engine.add_memory("Machine learning basics")

        # Search for programming-related content
        results = memory_engine.search_memories("programming", k=2)

        assert len(results) <= 2
        assert all(isinstance(r, Memory) for r in results)
        assert all(hasattr(r, "relevance_score") for r in results)

    def test_get_recent_memories(self, memory_engine):
        """Test getting recent memories in chronological order"""
        # Add memories with delay to ensure different timestamps
        memory1 = memory_engine.add_memory("First memory")
        memory2 = memory_engine.add_memory("Second memory")
        memory3 = memory_engine.add_memory("Third memory")

        recent = memory_engine.get_recent_memories(n=2)

        assert len(recent) == 2
        assert recent[0] == memory3  # Most recent first
        assert recent[1] == memory2

    def test_clear_memories(self, memory_engine):
        """Test clearing all memories"""
        memory_engine.add_memory("Test memory 1")
        memory_engine.add_memory("Test memory 2")

        assert len(memory_engine.memories) == 2

        memory_engine.clear_memories()

        assert len(memory_engine.memories) == 0

    def test_save_memories(self, memory_engine, temp_file):
        """Test saving memories to file"""
        memory_engine.add_memory("Test memory", {"type": "test"})

        # Manually trigger save (normally automatic)
        memory_engine.save_memories()

        assert os.path.exists(temp_file)

        with open(temp_file, "r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["content"] == "Test memory"
        assert data[0]["metadata"] == {"type": "test"}

    def test_load_memories(self, mock_embedding_provider, temp_file):
        """Test loading memories from file"""
        # Create test data
        test_data = [
            {
                "content": "Loaded memory",
                "metadata": {"source": "file"},
                "timestamp": "2023-01-01T12:00:00",
                "relevance_score": 0.5,
            }
        ]

        with open(temp_file, "w") as f:
            json.dump(test_data, f)

        # Create new engine that should load the data
        faiss_store = FaissVectorStore(dimension=10)
        engine = MemoryEngine(
            vector_store=faiss_store,
            embedding_provider=mock_embedding_provider,
            persist_path=temp_file,
        )

        assert len(engine.memories) == 1
        assert engine.memories[0].content == "Loaded memory"
        assert engine.memories[0].metadata == {"source": "file"}

    def test_load_memories_with_corrupted_file(
        self, mock_embedding_provider, temp_file
    ):
        """Test handling of corrupted persistence file"""
        # Create corrupted JSON file
        with open(temp_file, "w") as f:
            f.write("invalid json content")

        # Should handle gracefully and start with empty memories
        faiss_store = FaissVectorStore(dimension=10)
        engine = MemoryEngine(
            vector_store=faiss_store,
            embedding_provider=mock_embedding_provider,
            persist_path=temp_file,
        )

        assert len(engine.memories) == 0

    def test_load_memories_nonexistent_file(self, mock_embedding_provider):
        """Test loading when persistence file doesn't exist"""
        engine = MemoryEngine(
            embedding_provider=mock_embedding_provider,
            persist_path="/nonexistent/path/memories.json",
        )

        assert len(engine.memories) == 0

    def test_persistence_integration(self, mock_embedding_provider, temp_file):
        """Test full persistence integration"""
        # Create engine and add memories
        faiss_store1 = FaissVectorStore(dimension=10)
        engine1 = MemoryEngine(
            vector_store=faiss_store1,
            embedding_provider=mock_embedding_provider,
            persist_path=temp_file,
        )

        engine1.add_memory("Persistent memory 1")
        engine1.add_memory("Persistent memory 2")

        # Create new engine that should load the memories
        faiss_store2 = FaissVectorStore(dimension=10)
        engine2 = MemoryEngine(
            vector_store=faiss_store2,
            embedding_provider=mock_embedding_provider,
            persist_path=temp_file,
        )

        assert len(engine2.memories) == 2
        contents = [m.content for m in engine2.memories]
        assert "Persistent memory 1" in contents
        assert "Persistent memory 2" in contents

    def test_concurrent_save_memories(self, temp_dir):
        """Simulate concurrent saves and ensure data integrity"""
        persist_path = os.path.join(temp_dir, "memories.json")

        engines = []
        contents = []
        for i in range(5):
            engine = MemoryEngine()
            content = f"Concurrent memory {i}"
            engine.add_memory(content)
            engine.persist_path = persist_path
            engines.append(engine)
            contents.append(content)

        threads = [threading.Thread(target=e.save_memories) for e in engines]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(persist_path, "r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["content"] in contents
        assert not os.path.exists(persist_path + ".tmp")

    def test_identity_corrections_search(self, memory_engine):
        """Identity corrections should be returned regardless of relevance."""
        correction = memory_engine.add_identity_correction("User's name is Alice")
        other_memory = memory_engine.add_memory("Some unrelated memory about Python")

        # Simulate vector store search returning only the unrelated memory
        with patch.object(
            memory_engine.vector_store, "search", return_value=[other_memory]
        ):
            results = memory_engine.search_memories("irrelevant query", k=1)

        # Ensure the correction is stored and retrievable
        corrections = memory_engine.get_identity_corrections()
        assert correction in corrections

        # Identity correction should be included even though it wasn't in search results
        assert correction in results

    def test_multi_query_search_memories(self, memory_engine):
        """Multi-query search should aggregate results from query variations."""
        memory_engine.add_memory("Python programming tutorial")
        memory_engine.add_memory("Java programming tutorial")
        memory_engine.add_memory("Cooking recipes")

        def generator(query, n):
            return [
                "Python programming tutorial",
                "Java programming tutorial",
            ]

        results = memory_engine.multi_query_search_memories(
            "programming help", k=5, query_generator=generator
        )

        contents = [m.content for m in results]
        assert "Python programming tutorial" in contents
        assert "Java programming tutorial" in contents
