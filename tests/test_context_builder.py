import pytest
from unittest.mock import Mock
from core.context_builder import ContextBuilder
from core.memory_engine import Memory, MemoryEngine


class TestContextBuilder:
    """Test cases for the ContextBuilder class"""

    def test_context_builder_creation(self):
        """Test basic context builder creation"""
        memory_engine = MemoryEngine()
        builder = ContextBuilder(memory_engine)

        assert builder.memory_engine == memory_engine
        assert builder.max_context_length == 4000

    def test_context_builder_custom_length(self):
        """Test context builder with custom max length"""
        memory_engine = MemoryEngine()
        builder = ContextBuilder(memory_engine, max_context_length=2000)

        assert builder.max_context_length == 2000

    def test_build_context_empty_engine(self):
        """Test building context when memory engine is empty"""
        memory_engine = MemoryEngine()
        builder = ContextBuilder(memory_engine)

        context = builder.build_context()

        assert context == ""

    def test_build_context_recent_only(self):
        """Test building context with only recent memories"""
        memory_engine = MemoryEngine()
        memory_engine.add_memory("First memory")
        memory_engine.add_memory("Second memory")

        builder = ContextBuilder(memory_engine)
        context = builder.build_context(include_recent=2, include_relevant=0)

        assert "## Recent Context:" in context
        assert "Second memory" in context
        assert "First memory" in context
        assert "## Relevant Context:" not in context

    def test_build_context_with_query(self, memory_engine):
        """Test building context with query for relevant memories"""
        # Add some memories
        memory_engine.add_memory("Python programming tutorial")
        memory_engine.add_memory("JavaScript web development")
        memory_engine.add_memory("Coffee brewing guide")

        builder = ContextBuilder(memory_engine)
        context = builder.build_context(
            query="programming", include_recent=2, include_relevant=2
        )

        assert "## Recent Context:" in context
        assert "## Relevant Context:" in context
        assert "relevance:" in context  # Should show relevance scores

    def test_build_context_no_recent(self):
        """Test building context without recent memories"""
        memory_engine = MemoryEngine()
        memory_engine.add_memory("Test memory")

        builder = ContextBuilder(memory_engine)
        context = builder.build_context(include_recent=0, include_relevant=0)

        assert context == ""

    def test_build_context_truncation(self):
        """Test context truncation when exceeding max length"""
        memory_engine = MemoryEngine()

        # Add a very long memory that exceeds max context length
        long_content = "Very long memory content. " * 1000  # Very long string
        memory_engine.add_memory(long_content)

        builder = ContextBuilder(memory_engine, max_context_length=100)
        context = builder.build_context(include_recent=1)

        assert len(context) <= 103  # 100 + "..." (3 chars)
        assert context.endswith("...")

    def test_build_context_no_truncation_needed(self):
        """Test that short context is not truncated"""
        memory_engine = MemoryEngine()
        memory_engine.add_memory("Short memory")

        builder = ContextBuilder(memory_engine, max_context_length=1000)
        context = builder.build_context(include_recent=1)

        assert not context.endswith("...")
        assert "Short memory" in context

    def test_format_for_prompt(self):
        """Test formatting context for prompt"""
        memory_engine = MemoryEngine()
        builder = ContextBuilder(memory_engine)

        context = "Test context content"
        query = "Test query"

        formatted = builder.format_for_prompt(context, query)

        assert "Based on the following context" in formatted
        assert "Test context content" in formatted
        assert "Question: Test query" in formatted
        assert "Answer:" in formatted

    def test_build_context_relevance_scores(self, memory_engine):
        """Test that relevance scores are included in context"""
        # Add memories
        memory_engine.add_memory("Python programming")
        memory_engine.add_memory("Data science")

        builder = ContextBuilder(memory_engine)
        context = builder.build_context(
            query="programming", include_recent=0, include_relevant=2
        )

        # Should contain relevance scores
        assert "relevance:" in context
        # Should be formatted as decimal
        assert "0." in context or "1." in context

    def test_build_context_mixed_sections(self, memory_engine):
        """Test building context with both recent and relevant sections"""
        # Add several memories
        for i in range(5):
            memory_engine.add_memory(f"Memory {i}")

        builder = ContextBuilder(memory_engine)
        context = builder.build_context(
            query="test", include_recent=2, include_relevant=2
        )

        lines = context.split("\n")

        # Should have both sections
        recent_index = next(
            i for i, line in enumerate(lines) if "## Recent Context:" in line
        )
        relevant_index = next(
            i for i, line in enumerate(lines) if "## Relevant Context:" in line
        )

        # Recent should come before relevant
        assert recent_index < relevant_index

    @pytest.fixture
    def mock_memory_engine_with_search(self):
        """Create a mock memory engine with search results"""
        mock_engine = Mock()

        # Mock recent memories
        mock_engine.get_recent_memories.return_value = [
            Memory(content="Recent memory 1"),
            Memory(content="Recent memory 2"),
        ]

        # Mock search results with relevance scores
        search_results = [
            Memory(content="Relevant memory 1", relevance_score=0.95),
            Memory(content="Relevant memory 2", relevance_score=0.85),
        ]
        mock_engine.search_memories.return_value = search_results

        return mock_engine

    def test_build_context_with_mocked_engine(self, mock_memory_engine_with_search):
        """Test context building with mocked memory engine"""
        builder = ContextBuilder(mock_memory_engine_with_search)
        context = builder.build_context(
            query="test query", include_recent=2, include_relevant=2
        )

        assert "Recent memory 1" in context
        assert "Recent memory 2" in context
        assert "Relevant memory 1" in context
        assert "Relevant memory 2" in context
        assert "0.95" in context  # Relevance score
        assert "0.85" in context  # Relevance score

        # Verify method calls
        mock_memory_engine_with_search.get_recent_memories.assert_called_once_with(2)
        mock_memory_engine_with_search.search_memories.assert_called_once_with(
            "test query", 2
        )

    def test_profile_memories_included(self):
        """Profile memories should be included when a profile query is provided"""
        engine = MemoryEngine()
        mem = engine.add_memory("Jeremy likes skiing")
        mem.relevance_score = 0.9

        builder = ContextBuilder(engine, profile_query="Jeremy")
        context = builder.build_context(include_recent=0, include_relevant=0)

        assert "## Profile Information:" in context
        assert "Jeremy likes skiing" in context

    def test_profile_memories_excluded_without_query(self):
        """Profile memories are skipped when no profile query is supplied"""
        engine = MemoryEngine()
        mem = engine.add_memory("Jeremy likes skiing")
        mem.relevance_score = 0.9

        builder = ContextBuilder(engine)
        context = builder.build_context(include_recent=0, include_relevant=0)

        assert context == ""
