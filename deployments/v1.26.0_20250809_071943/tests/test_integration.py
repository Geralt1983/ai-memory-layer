import pytest
import tempfile
import os
from unittest.mock import patch, Mock
from core.memory_engine import MemoryEngine
from storage.faiss_store import FaissVectorStore
from integrations.embeddings import OpenAIEmbeddings
from integrations.openai_integration import OpenAIIntegration
from tests.conftest import MockEmbeddingProvider


@pytest.mark.integration
class TestFullIntegration:
    """Integration tests for the complete AI memory layer system"""

    def test_end_to_end_memory_workflow(self):
        """Test complete workflow: add memories, search, persist, reload"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup components
            embedding_provider = MockEmbeddingProvider()
            vector_store = FaissVectorStore(
                dimension=10, index_path=os.path.join(temp_dir, "test_index")
            )
            memory_persist_path = os.path.join(temp_dir, "memories.json")

            # Create memory engine
            engine = MemoryEngine(
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                persist_path=memory_persist_path,
            )

            # Add some memories
            engine.add_memory("Python is a programming language", {"type": "fact"})
            engine.add_memory("I love coffee in the morning", {"type": "preference"})
            engine.add_memory("Machine learning uses algorithms", {"type": "fact"})

            assert len(engine.memories) == 3

            # Test search
            results = engine.search_memories("programming", k=2)
            assert len(results) <= 2
            assert all(hasattr(r, "relevance_score") for r in results)

            # Test recent memories
            recent = engine.get_recent_memories(n=2)
            assert len(recent) == 2
            assert (
                recent[0].content == "Machine learning uses algorithms"
            )  # Most recent

            # Verify persistence files exist
            assert os.path.exists(memory_persist_path)
            assert os.path.exists(f"{vector_store.index_path}.index")
            assert os.path.exists(f"{vector_store.index_path}.pkl")

            # Create new engine and verify it loads everything
            new_vector_store = FaissVectorStore(
                dimension=10, index_path=os.path.join(temp_dir, "test_index")
            )
            new_engine = MemoryEngine(
                vector_store=new_vector_store,
                embedding_provider=embedding_provider,
                persist_path=memory_persist_path,
            )

            assert len(new_engine.memories) == 3
            assert new_vector_store.index.ntotal == 3

            # Test search on reloaded engine
            new_results = new_engine.search_memories("coffee", k=1)
            assert len(new_results) == 1

    @patch("integrations.openai_integration.OpenAI")
    def test_openai_integration_workflow(self, mock_openai_class):
        """Test OpenAI integration with memory workflow"""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock()]
        mock_chat_response.choices[0].message.content = (
            "I understand you're asking about Python!"
        )
        mock_client.chat.completions.create.return_value = mock_chat_response

        # Setup memory system
        embedding_provider = MockEmbeddingProvider()
        vector_store = FaissVectorStore(dimension=10)
        memory_engine = MemoryEngine(
            vector_store=vector_store, embedding_provider=embedding_provider
        )

        # Create OpenAI integration
        with patch(
            "integrations.openai_integration.OpenAIEmbeddings"
        ) as mock_embeddings:
            mock_embeddings.return_value = embedding_provider

            ai = OpenAIIntegration(
                api_key="test-key", memory_engine=memory_engine, model="gpt-3.5-turbo"
            )

            # Add some background knowledge
            ai.add_memory_with_embedding("Python is a versatile programming language")
            ai.add_memory_with_embedding("Python is great for data science")

            # Have a conversation
            response = ai.chat_with_memory(
                "Tell me about Python programming",
                system_prompt="You are a helpful programming assistant",
            )

            assert response == "I understand you're asking about Python!"

            # Verify conversation was stored
            user_messages = [m for m in memory_engine.memories if "User:" in m.content]
            ai_messages = [
                m for m in memory_engine.memories if "Assistant:" in m.content
            ]

            assert len(user_messages) == 1
            assert len(ai_messages) == 1
            assert "Tell me about Python programming" in user_messages[0].content

            # Verify total memories (2 background + 1 user + 1 ai)
            assert len(memory_engine.memories) == 4

    def test_context_building_integration(self):
        """Test that context building works with real memory search"""
        embedding_provider = MockEmbeddingProvider()
        vector_store = FaissVectorStore(dimension=10)
        memory_engine = MemoryEngine(
            vector_store=vector_store, embedding_provider=embedding_provider
        )

        # Add memories with different topics
        memory_engine.add_memory("Python tutorial: variables and functions")
        memory_engine.add_memory("JavaScript basics: DOM manipulation")
        memory_engine.add_memory("Python advanced: decorators and generators")
        memory_engine.add_memory("Coffee brewing techniques")
        memory_engine.add_memory("Python libraries: numpy and pandas")

        # Import here to avoid circular dependency
        from core.context_builder import ContextBuilder

        builder = ContextBuilder(memory_engine, max_context_length=1000)

        # Build context for Python-related query
        context = builder.build_context(
            query="Python programming", include_recent=2, include_relevant=3
        )

        # Should contain both recent and relevant sections
        assert "## Recent Context:" in context
        assert "## Relevant Context:" in context

        # Should contain Python-related content
        assert "Python" in context
        assert "relevance:" in context

        # Should not exceed max length
        assert len(context) <= 1000

    def test_memory_cleanup_integration(self):
        """Test memory cleanup functionality"""
        embedding_provider = MockEmbeddingProvider()
        vector_store = FaissVectorStore(dimension=10)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            persist_path = f.name

        try:
            memory_engine = MemoryEngine(
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                persist_path=persist_path,
            )

            # Add memories
            memory_engine.add_memory("Memory 1")
            memory_engine.add_memory("Memory 2")
            memory_engine.add_memory("Memory 3")

            assert len(memory_engine.memories) == 3
            assert vector_store.index.ntotal == 3
            assert os.path.exists(persist_path)

            # Clear memories
            memory_engine.clear_memories()

            assert len(memory_engine.memories) == 0

            # Verify persistence file is updated (should be empty array)
            import json

            with open(persist_path, "r") as f:
                data = json.load(f)
            assert data == []

        finally:
            # Cleanup
            if os.path.exists(persist_path):
                os.unlink(persist_path)

    def test_error_handling_integration(self):
        """Test error handling in integrated system"""
        embedding_provider = MockEmbeddingProvider()

        # Test with invalid vector store path
        vector_store = FaissVectorStore(dimension=10, index_path="/invalid/path/index")

        memory_engine = MemoryEngine(
            vector_store=vector_store, embedding_provider=embedding_provider
        )

        # Should still work for basic operations
        memory = memory_engine.add_memory("Test memory")
        assert memory.content == "Test memory"
        assert len(memory_engine.memories) == 1

        # Search should work (fallback to recent memories if vector store fails)
        results = memory_engine.search_memories("test", k=1)
        assert len(results) == 1
