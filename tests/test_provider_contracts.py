"""Tests for provider contract enforcement."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Sequence

from memory_layer.providers.base import EmbeddingProvider, EmbeddingConfig, EmbeddingError, ProviderUnavailable
from memory_layer.providers.openai import OpenAIEmbeddings
from memory_layer.providers.voyage import VoyageEmbeddings


class ContractTestProvider(EmbeddingProvider):
    """Test provider for contract validation."""
    
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__(cfg)
        self.available = True
        self.call_count = 0
        self.batch_sizes = []
        self.timeout_used = None
    
    def is_available(self) -> bool:
        return self.available
    
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        self.call_count += 1
        self.batch_sizes.append(len(texts))
        # Simulate respecting max_batch_size by raising if exceeded
        if len(texts) > self.cfg.max_batch_size:
            raise EmbeddingError(f"Batch size {len(texts)} exceeds limit {self.cfg.max_batch_size}")
        return [[0.1 * i, 0.2 * i] for i in range(len(texts))]


@pytest.mark.unit
class TestProviderContracts:
    """Test that providers respect their contracts."""
    
    def test_max_batch_size_respected(self):
        """Test that providers don't exceed max_batch_size."""
        cfg = EmbeddingConfig(model="test", dim=2, max_batch_size=3)
        provider = ContractTestProvider(cfg)
        
        # Test with exactly max_batch_size
        texts = ["text1", "text2", "text3"]
        result = provider.embed_batch(texts)
        assert len(result) == 3
        
        # Test that larger batch raises error
        large_texts = ["text" + str(i) for i in range(5)]
        with pytest.raises(EmbeddingError, match="exceeds limit"):
            provider.embed_batch(large_texts)
    
    def test_timeout_parameter_available(self):
        """Test that providers have timeout_s in config."""
        cfg = EmbeddingConfig(model="test", dim=2, timeout_s=15.0)
        provider = ContractTestProvider(cfg)
        assert provider.cfg.timeout_s == 15.0
    
    def test_max_retries_parameter(self):
        """Test that providers have max_retries in config."""
        cfg = EmbeddingConfig(model="test", dim=2, max_retries=5)
        provider = ContractTestProvider(cfg)
        assert provider.cfg.max_retries == 5
    
    def test_is_available_contract(self):
        """Test that is_available() returns boolean."""
        cfg = EmbeddingConfig(model="test", dim=2)
        provider = ContractTestProvider(cfg)
        
        result = provider.is_available()
        assert isinstance(result, bool)
        
        provider.available = False
        result = provider.is_available()
        assert result is False
    
    def test_embed_batch_returns_correct_structure(self):
        """Test that embed_batch returns List[List[float]]."""
        cfg = EmbeddingConfig(model="test", dim=2)
        provider = ContractTestProvider(cfg)
        
        texts = ["hello", "world"]
        result = provider.embed_batch(texts)
        
        assert isinstance(result, list)
        assert len(result) == len(texts)
        for embedding in result:
            assert isinstance(embedding, list)
            for value in embedding:
                assert isinstance(value, (int, float))
    
    def test_empty_texts_handling(self):
        """Test that providers handle empty input gracefully."""
        cfg = EmbeddingConfig(model="test", dim=2)
        provider = ContractTestProvider(cfg)
        
        result = provider.embed_batch([])
        assert result == []


@pytest.mark.unit
class TestOpenAIContractCompliance:
    """Test OpenAI provider contract compliance."""
    
    def test_retry_with_jitter_implementation(self):
        """Test that OpenAI provider implements retry with jitter."""
        cfg = EmbeddingConfig(model="text-embedding-3-small", dim=1536, max_retries=2)
        provider = OpenAIEmbeddings(cfg)
        
        # Mock requests to simulate failures
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server error")
        
        with patch.object(provider.session, 'post', return_value=mock_response):
            with patch('time.sleep') as mock_sleep:
                with patch('random.uniform', return_value=0.3):  # Fixed jitter for testing
                    
                    with pytest.raises(Exception):
                        provider.embed_batch(["test"])
                    
                    # Should have called sleep with jitter
                    assert mock_sleep.call_count > 0
                    # First retry should include jitter
                    first_sleep_call = mock_sleep.call_args_list[0][0][0]
                    assert first_sleep_call > 1.0  # base delay + jitter
    
    def test_respects_max_batch_size(self):
        """Test that OpenAI provider respects max_batch_size."""
        cfg = EmbeddingConfig(model="test", dim=1536, max_batch_size=2)
        provider = OpenAIEmbeddings(cfg)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
        }
        
        with patch.object(provider.session, 'post', return_value=mock_response) as mock_post:
            texts = ["text1", "text2", "text3", "text4"]  # 4 texts, batch size 2
            provider.embed_batch(texts)
            
            # Should make 2 calls due to batch size limit
            assert mock_post.call_count == 2
            
            # First call should have 2 texts
            first_call_data = mock_post.call_args_list[0][1]['json']
            assert len(first_call_data['input']) == 2
            
            # Second call should have 2 texts  
            second_call_data = mock_post.call_args_list[1][1]['json']
            assert len(second_call_data['input']) == 2
    
    def test_timeout_passed_to_requests(self):
        """Test that timeout_s is passed to requests."""
        cfg = EmbeddingConfig(model="test", dim=1536, timeout_s=30.0)
        provider = OpenAIEmbeddings(cfg)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1]}]}
        
        with patch.object(provider.session, 'post', return_value=mock_response) as mock_post:
            provider.embed_batch(["test"])
            
            # Should pass timeout to requests
            mock_post.assert_called_with(
                "https://api.openai.com/v1/embeddings",
                json={"model": "test", "input": ["test"]},
                timeout=30.0
            )


@pytest.mark.unit  
class TestVoyageContractCompliance:
    """Test Voyage provider contract compliance."""
    
    def test_retry_with_jitter_implementation(self):
        """Test that Voyage provider implements retry with jitter."""
        cfg = EmbeddingConfig(model="voyage-2", dim=1536, max_retries=2)
        provider = VoyageEmbeddings(cfg)
        
        # Mock requests to simulate failures
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server error")
        
        with patch.object(provider.session, 'post', return_value=mock_response):
            with patch('time.sleep') as mock_sleep:
                with patch('random.uniform', return_value=0.2):  # Fixed jitter for testing
                    
                    with pytest.raises(Exception):
                        provider.embed_batch(["test"])
                    
                    # Should have called sleep with jitter
                    assert mock_sleep.call_count > 0
                    # First retry should include jitter  
                    first_sleep_call = mock_sleep.call_args_list[0][0][0]
                    assert first_sleep_call > 0.5  # base delay + jitter
    
    def test_respects_max_batch_size(self):
        """Test that Voyage provider respects max_batch_size."""
        cfg = EmbeddingConfig(model="voyage-2", dim=1536, max_batch_size=3)
        provider = VoyageEmbeddings(cfg)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]} for _ in range(3)]
        }
        
        with patch.object(provider.session, 'post', return_value=mock_response) as mock_post:
            texts = ["text" + str(i) for i in range(7)]  # 7 texts, batch size 3
            provider.embed_batch(texts)
            
            # Should make 3 calls: 3+3+1 due to batch size limit
            assert mock_post.call_count == 3
            
            # Check batch sizes
            call_inputs = [call[1]['json']['input'] for call in mock_post.call_args_list]
            assert len(call_inputs[0]) == 3  # First batch
            assert len(call_inputs[1]) == 3  # Second batch  
            assert len(call_inputs[2]) == 1  # Final batch
    
    def test_provider_unavailable_when_no_key(self):
        """Test that provider reports unavailable when API key missing."""
        cfg = EmbeddingConfig(model="voyage-2", dim=1536)
        
        with patch.dict('os.environ', {}, clear=True):
            provider = VoyageEmbeddings(cfg)
            assert not provider.is_available()
            
            with pytest.raises(ProviderUnavailable, match="VOYAGE_API_KEY not set"):
                provider.embed_batch(["test"])


@pytest.mark.integration
class TestProviderIntegrationContracts:
    """Integration tests for provider contracts."""
    
    def test_openai_provider_contract_integration(self):
        """Test OpenAI provider through full config flow."""
        from memory_layer.config import AppConfig, build_provider
        
        # Test that config produces provider with correct contract
        app_cfg = AppConfig(
            provider="openai", 
            model="text-embedding-3-small",
            max_batch_size=16,
            timeout_s=20.0,
            max_retries=3
        )
        
        try:
            provider = build_provider(app_cfg)
            assert hasattr(provider, 'is_available')
            assert hasattr(provider, 'embed_batch')
            assert provider.cfg.max_batch_size == 16
            assert provider.cfg.timeout_s == 20.0
            assert provider.cfg.max_retries == 3
        except ValueError:
            pytest.skip("OpenAI provider not available in registry")
    
    def test_voyage_provider_contract_integration(self):
        """Test Voyage provider through full config flow."""
        from memory_layer.config import AppConfig, build_provider
        
        app_cfg = AppConfig(
            provider="voyage",
            model="voyage-2", 
            max_batch_size=8,
            timeout_s=15.0,
            max_retries=2
        )
        
        provider = build_provider(app_cfg)
        assert hasattr(provider, 'is_available') 
        assert hasattr(provider, 'embed_batch')
        assert provider.cfg.max_batch_size == 8
        assert provider.cfg.timeout_s == 15.0
        assert provider.cfg.max_retries == 2