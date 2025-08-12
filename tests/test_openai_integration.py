import pytest
from unittest.mock import Mock, patch, MagicMock
from integrations.openai_integration import OpenAIIntegration
from core.memory_engine import MemoryEngine


class TestOpenAIIntegration:
    """Test cases for OpenAI integration"""

    @patch("integrations.openai_integration.OpenAI")
    @patch("integrations.openai_integration.OpenAIEmbeddings")
    @patch("integrations.openai_integration.ContextBuilder")
    def test_openai_integration_creation(
        self, mock_context_builder, mock_embeddings, mock_openai
    ):
        """Test OpenAI integration initialization"""
        memory_engine = MemoryEngine()

        integration = OpenAIIntegration(
            api_key="test-key",
            memory_engine=memory_engine,
            model="gpt-4",
            embedding_model="text-embedding-3-small",
        )

        assert integration.model == "gpt-4"
        assert integration.memory_engine == memory_engine
        mock_openai.assert_called_once_with(api_key="test-key")
        mock_embeddings.assert_called_once_with("test-key", "text-embedding-3-small")
        mock_context_builder.assert_called_once_with(
            memory_engine, profile_query="Jeremy wife Ashley kids dogs age"
        )

    @patch("integrations.openai_integration.OpenAI")
    @patch("integrations.openai_integration.OpenAIEmbeddings")
    @patch("integrations.openai_integration.ContextBuilder")
    def test_chat_with_memory_basic(
        self, mock_context_builder, mock_embeddings, mock_openai
    ):
        """Test basic chat with memory functionality"""
        # Setup mocks
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI response"
        mock_client.chat.completions.create.return_value = mock_response

        mock_builder = Mock()
        mock_builder.build_context.return_value = "Context from memories"
        mock_context_builder.return_value = mock_builder

        memory_engine = MemoryEngine()
        integration = OpenAIIntegration(api_key="test-key", memory_engine=memory_engine)

        # Test chat
        response = integration.chat_with_memory("Hello AI")

        assert response == "AI response"
        mock_builder.build_context.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()

    @patch("integrations.openai_integration.OpenAI")
    @patch("integrations.openai_integration.OpenAIEmbeddings")
    @patch("integrations.openai_integration.ContextBuilder")
    def test_chat_with_system_prompt(
        self, mock_context_builder, mock_embeddings, mock_openai
    ):
        """Test chat with custom system prompt"""
        # Setup mocks
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI response"
        mock_client.chat.completions.create.return_value = mock_response

        mock_builder = Mock()
        mock_builder.build_context.return_value = ""
        mock_context_builder.return_value = mock_builder

        memory_engine = MemoryEngine()
        integration = OpenAIIntegration(api_key="test-key", memory_engine=memory_engine)

        # Test chat with system prompt
        response = integration.chat_with_memory(
            "Hello", system_prompt="You are a helpful assistant"
        )

        # Check that system prompt was included in messages
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) >= 1
        assert any("helpful assistant" in m["content"] for m in system_messages)

    @patch("integrations.openai_integration.OpenAI")
    @patch("integrations.openai_integration.OpenAIEmbeddings")
    @patch("integrations.openai_integration.ContextBuilder")
    def test_chat_with_context(
        self, mock_context_builder, mock_embeddings, mock_openai
    ):
        """Test chat with memory context"""
        # Setup mocks
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI response"
        mock_client.chat.completions.create.return_value = mock_response

        mock_builder = Mock()
        mock_builder.build_context.return_value = "Previous conversation context"
        mock_context_builder.return_value = mock_builder

        memory_engine = MemoryEngine()
        integration = OpenAIIntegration(api_key="test-key", memory_engine=memory_engine)

        # Test chat
        integration.chat_with_memory("Hello")

        # Check that context was included in messages
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        context_messages = [
            m for m in messages if "Previous context:" in m.get("content", "")
        ]
        assert len(context_messages) == 1
        assert "Previous conversation context" in context_messages[0]["content"]

    @patch("integrations.openai_integration.OpenAI")
    @patch("integrations.openai_integration.OpenAIEmbeddings")
    @patch("integrations.openai_integration.ContextBuilder")
    def test_chat_remember_response(
        self, mock_context_builder, mock_embeddings, mock_openai
    ):
        """Test that conversation is stored in memory"""
        # Setup mocks
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI response"
        mock_client.chat.completions.create.return_value = mock_response

        mock_builder = Mock()
        mock_builder.build_context.return_value = ""
        mock_context_builder.return_value = mock_builder

        memory_engine = MemoryEngine()
        integration = OpenAIIntegration(api_key="test-key", memory_engine=memory_engine)

        # Test chat with memory storage
        integration.chat_with_memory("Hello AI", remember_response=True)

        # Check that memories were added
        assert len(memory_engine.memories) == 2  # User message + AI response

        user_memory = next(
            (m for m in memory_engine.memories if "User:" in m.content), None
        )
        ai_memory = next(
            (m for m in memory_engine.memories if "Assistant:" in m.content), None
        )

        assert user_memory is not None
        assert "Hello AI" in user_memory.content
        assert user_memory.metadata["type"] == "user_message"

        assert ai_memory is not None
        assert "AI response" in ai_memory.content
        assert ai_memory.metadata["type"] == "assistant_response"

    @patch("integrations.openai_integration.OpenAI")
    @patch("integrations.openai_integration.OpenAIEmbeddings")
    @patch("integrations.openai_integration.ContextBuilder")
    def test_chat_no_remember(self, mock_context_builder, mock_embeddings, mock_openai):
        """Test chat without storing in memory"""
        # Setup mocks
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI response"
        mock_client.chat.completions.create.return_value = mock_response

        mock_builder = Mock()
        mock_builder.build_context.return_value = ""
        mock_context_builder.return_value = mock_builder

        memory_engine = MemoryEngine()
        integration = OpenAIIntegration(api_key="test-key", memory_engine=memory_engine)

        # Test chat without memory storage
        integration.chat_with_memory("Hello AI", remember_response=False)

        # Check that no memories were added
        assert len(memory_engine.memories) == 0

    @patch("integrations.openai_integration.OpenAI")
    @patch("integrations.openai_integration.OpenAIEmbeddings")
    def test_add_memory_with_embedding(self, mock_embeddings, mock_openai):
        """Test adding memory with embedding"""
        memory_engine = MemoryEngine()
        integration = OpenAIIntegration(api_key="test-key", memory_engine=memory_engine)

        # Test adding memory
        memory = integration.add_memory_with_embedding(
            "Test memory content", metadata={"source": "test"}
        )

        assert memory.content == "Test memory content"
        assert memory.metadata == {"source": "test"}
        assert len(memory_engine.memories) == 1

    @patch("integrations.openai_integration.OpenAI")
    @patch("integrations.openai_integration.OpenAIEmbeddings")
    @patch("integrations.openai_integration.ContextBuilder")
    def test_context_builder_parameters(
        self, mock_context_builder, mock_embeddings, mock_openai
    ):
        """Test that context builder receives correct parameters"""
        # Setup mocks
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI response"
        mock_client.chat.completions.create.return_value = mock_response

        mock_builder = Mock()
        mock_builder.build_context.return_value = ""
        mock_context_builder.return_value = mock_builder

        memory_engine = MemoryEngine()
        integration = OpenAIIntegration(api_key="test-key", memory_engine=memory_engine)

        # Test chat with custom parameters
        integration.chat_with_memory(
            "Test message", include_recent=3, include_relevant=7
        )

        # Check that context builder was called with correct parameters
        mock_builder.build_context.assert_called_once_with(
            query="Test message", include_recent=3, include_relevant=7
        )
