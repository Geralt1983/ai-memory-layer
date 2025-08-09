import pytest
import numpy as np
from unittest.mock import Mock, patch
from integrations.embeddings import OpenAIEmbeddings


class TestOpenAIEmbeddings:
    """Test cases for OpenAI embeddings provider"""

    @patch("integrations.embeddings.OpenAI")
    def test_openai_embeddings_creation(self, mock_openai_class):
        """Test OpenAI embeddings provider creation"""
        provider = OpenAIEmbeddings("test-api-key", "text-embedding-ada-002")

        assert provider.model == "text-embedding-ada-002"
        mock_openai_class.assert_called_once_with(api_key="test-api-key")

    @patch("integrations.embeddings.OpenAI")
    def test_embed_text_string(self, mock_openai_class):
        """Test embedding a single string"""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIEmbeddings("test-api-key")
        result = provider.embed_text("test text")

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([0.1, 0.2, 0.3]))
        mock_client.embeddings.create.assert_called_once_with(
            input="test text", model="text-embedding-ada-002"
        )

    @patch("integrations.embeddings.OpenAI")
    def test_embed_text_list(self, mock_openai_class):
        """Test embedding a list of strings (joins them)"""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIEmbeddings("test-api-key")
        result = provider.embed_text(["hello", "world"])

        assert isinstance(result, np.ndarray)
        mock_client.embeddings.create.assert_called_once_with(
            input="hello world", model="text-embedding-ada-002"
        )

    @patch("integrations.embeddings.OpenAI")
    def test_embed_batch(self, mock_openai_class):
        """Test batch embedding multiple texts"""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock(), Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_response.data[1].embedding = [0.4, 0.5, 0.6]
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIEmbeddings("test-api-key")
        texts = ["first text", "second text"]
        results = provider.embed_batch(texts)

        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)
        assert np.array_equal(results[0], np.array([0.1, 0.2, 0.3]))
        assert np.array_equal(results[1], np.array([0.4, 0.5, 0.6]))

        mock_client.embeddings.create.assert_called_once_with(
            input=texts, model="text-embedding-ada-002"
        )

    @patch("integrations.embeddings.OpenAI")
    def test_custom_model(self, mock_openai_class):
        """Test using a custom embedding model"""
        provider = OpenAIEmbeddings("test-api-key", "custom-model")

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2]
        mock_client.embeddings.create.return_value = mock_response

        provider.embed_text("test")

        mock_client.embeddings.create.assert_called_once_with(
            input="test", model="custom-model"
        )


class TestMockEmbeddingProvider:
    """Test cases for the mock embedding provider used in tests"""

    def test_mock_embedding_provider(self, mock_embedding_provider):
        """Test that mock embedding provider works correctly"""
        text = "test text"
        embedding = mock_embedding_provider.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 10
        assert embedding.dtype == np.float32

    def test_mock_embedding_consistency(self, mock_embedding_provider):
        """Test that mock embeddings are consistent for same input"""
        text = "consistent test"
        embedding1 = mock_embedding_provider.embed_text(text)
        embedding2 = mock_embedding_provider.embed_text(text)

        assert np.array_equal(embedding1, embedding2)

    def test_mock_embedding_different_texts(self, mock_embedding_provider):
        """Test that different texts produce different embeddings"""
        embedding1 = mock_embedding_provider.embed_text("text one")
        embedding2 = mock_embedding_provider.embed_text("text two")

        assert not np.array_equal(embedding1, embedding2)

    def test_mock_batch_embedding(self, mock_embedding_provider):
        """Test mock batch embedding functionality"""
        texts = ["first", "second", "third"]
        embeddings = mock_embedding_provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(len(e) == 10 for e in embeddings)
