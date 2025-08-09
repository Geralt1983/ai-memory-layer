"""Tests for OpenAI provider contract compliance."""

import os
import pytest
from unittest.mock import Mock, patch
from memory_layer.providers.base import EmbeddingConfig, ProviderUnavailable
from memory_layer.providers.openai import OpenAIEmbeddings

@pytest.mark.unit
class TestOpenAIProviderContract:
    """Test OpenAI provider follows the contract."""
    
    def test_openai_provider_unavailable_without_key(self):
        """Test that OpenAI provider reports unavailable without API key."""
        with patch.dict('os.environ', {}, clear=True):
            config = EmbeddingConfig(model="text-embedding-3-small", dim=1536)
            provider = OpenAIEmbeddings(config)
            
            assert provider.is_available() is False
            
            with pytest.raises(ProviderUnavailable, match="OPENAI_API_KEY not set"):
                provider.embed_batch(["test"])
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_openai_provider_available_with_key(self):
        """Test that OpenAI provider reports available with API key."""
        config = EmbeddingConfig(model="text-embedding-3-small", dim=1536)
        provider = OpenAIEmbeddings(config)
        
        assert provider.is_available() is True
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('requests.Session.post')
    def test_openai_embed_batch_success(self, mock_post):
        """Test successful embedding batch request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }
        mock_post.return_value = mock_response
        
        config = EmbeddingConfig(model="text-embedding-3-small", dim=3, normalize=False)
        provider = OpenAIEmbeddings(config)
        
        result = provider.embed_batch(["text1", "text2"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        
        # Check request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['model'] == "text-embedding-3-small"
        assert call_args[1]['json']['input'] == ["text1", "text2"]
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('requests.Session.post')
    def test_openai_embed_batch_with_normalization(self, mock_post):
        """Test embedding batch with L2 normalization."""
        # Mock response with non-normalized vectors
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [3.0, 4.0]}]  # magnitude = 5
        }
        mock_post.return_value = mock_response
        
        config = EmbeddingConfig(model="text-embedding-3-small", dim=2, normalize=True)
        provider = OpenAIEmbeddings(config)
        
        result = provider.embed_batch(["text"])
        
        # Should be normalized (3/5, 4/5) = (0.6, 0.8)
        assert len(result) == 1
        assert abs(result[0][0] - 0.6) < 1e-6
        assert abs(result[0][1] - 0.8) < 1e-6
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('requests.Session.post')
    def test_openai_embed_query_single_result(self, mock_post):
        """Test embed_query returns single vector."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response
        
        config = EmbeddingConfig(model="text-embedding-3-small", dim=3)
        provider = OpenAIEmbeddings(config)
        
        result = provider.embed_query("single text")
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('requests.Session.post')
    def test_openai_retry_logic(self, mock_post):
        """Test retry logic for 5xx and 429 errors."""
        # First call fails with 503, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 503
        mock_response_fail.raise_for_status.side_effect = Exception("Service unavailable")
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "data": [{"embedding": [0.1, 0.2]}]
        }
        
        mock_post.side_effect = [mock_response_fail, mock_response_success]
        
        config = EmbeddingConfig(model="text-embedding-3-small", dim=2, max_retries=2)
        provider = OpenAIEmbeddings(config)
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = provider.embed_batch(["test"])
        
        assert len(result) == 1
        assert result[0] == [0.1, 0.2]
        assert mock_post.call_count == 2
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_openai_empty_input_handling(self):
        """Test that empty input returns empty list."""
        config = EmbeddingConfig(model="text-embedding-3-small", dim=1536)
        provider = OpenAIEmbeddings(config)
        
        result = provider.embed_batch([])
        assert result == []
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key', 'OPENAI_ORG_ID': 'org-123'})
    def test_openai_organization_header(self):
        """Test that organization header is set when provided."""
        config = EmbeddingConfig(model="text-embedding-3-small", dim=1536)
        provider = OpenAIEmbeddings(config)
        
        assert provider.session.headers.get('OpenAI-Organization') == 'org-123'
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('requests.Session.post')
    def test_openai_batch_size_chunking(self, mock_post):
        """Test that large batches are chunked according to max_batch_size."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2]}]
        }
        mock_post.return_value = mock_response
        
        config = EmbeddingConfig(model="text-embedding-3-small", dim=2, max_batch_size=2)
        provider = OpenAIEmbeddings(config)
        
        # Try to embed 3 texts with batch size 2
        result = provider.embed_batch(["text1", "text2", "text3"])
        
        # Should make 2 API calls: one with 2 texts, one with 1 text
        assert mock_post.call_count == 2
        assert len(result) == 3
        
        # Check the chunks were correct
        first_call = mock_post.call_args_list[0][1]['json']['input']
        second_call = mock_post.call_args_list[1][1]['json']['input']
        assert len(first_call) == 2
        assert len(second_call) == 1


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
def test_openai_contract_roundtrip():
    """Integration test with real OpenAI API (if key available)."""
    config = EmbeddingConfig(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"), 
        dim=1536
    )
    provider = OpenAIEmbeddings(config)
    
    assert provider.is_available()
    
    # Test batch embedding
    vecs = provider.embed_batch(["alpha", "beta"])
    assert len(vecs) == 2
    assert len(vecs[0]) == provider.cfg.dim
    assert len(vecs[1]) == provider.cfg.dim
    
    # Test single query
    q = provider.embed_query("alpha")
    assert len(q) == provider.cfg.dim
    
    # Vectors should be different
    assert vecs[0] != vecs[1]