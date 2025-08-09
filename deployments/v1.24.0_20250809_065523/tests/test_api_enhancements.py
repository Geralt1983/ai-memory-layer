"""
Test enhanced FastAPI functionality including Pydantic models, webhook auth, and background tasks
"""

import pytest
import hmac
import hashlib
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

# Mock the memory engine and related imports to avoid dependency issues
mock_memory_engine = Mock()
mock_memory_engine.memories = []
mock_memory_engine.add_memory = Mock()
mock_memory_engine.search_memories = Mock(return_value=[])

with patch.dict('sys.modules', {
    'optimized_memory_loader': Mock(),
    'optimized_clean_loader': Mock(), 
    'core.gpt_response': Mock(),
    'core.similarity_utils': Mock(),
    'core.memory_synthesis': Mock(),
    'core.memory_manager': Mock()
}):
    with patch('chatgpt_memory_api.memory_engine', mock_memory_engine):
        with patch('chatgpt_memory_api.create_optimized_chatgpt_engine', return_value=mock_memory_engine):
            from chatgpt_memory_api import app


class TestAPIEnhancements:
    """Test enhanced FastAPI functionality"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
        mock_memory_engine.reset_mock()
    
    def test_health_endpoint_response_model(self):
        """Test health endpoint returns proper response model"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields from HealthResponse model
        assert "status" in data
        assert "memory_count" in data
        assert "system" in data
        assert "dataset_size" in data
        
        assert data["status"] == "healthy"
        assert isinstance(data["memory_count"], int)
    
    def test_chat_request_validation(self):
        """Test ChatRequest model validation"""
        # Valid request
        valid_request = {
            "message": "Hello world",
            "use_gpt": True,
            "context_k": 10
        }
        
        with patch('chatgpt_memory_api.generate_gpt_response') as mock_gpt:
            mock_gpt.return_value = "Test response"
            response = self.client.post("/chat", json=valid_request)
        
        assert response.status_code == 200
    
    def test_chat_request_validation_failures(self):
        """Test ChatRequest validation failures"""
        # Empty message
        response = self.client.post("/chat", json={"message": ""})
        assert response.status_code == 422  # Validation error
        
        # Message too long
        long_message = "x" * 10001
        response = self.client.post("/chat", json={"message": long_message})
        assert response.status_code == 422
        
        # Invalid context_k
        response = self.client.post("/chat", json={
            "message": "test",
            "context_k": -1
        })
        assert response.status_code == 422
        
        response = self.client.post("/chat", json={
            "message": "test", 
            "context_k": 100  # Above limit
        })
        assert response.status_code == 422
    
    def test_search_request_validation(self):
        """Test SearchRequest model validation"""
        # Valid request
        response = self.client.post("/memories/search", json={
            "query": "test query",
            "k": 5,
            "include_scores": True
        })
        assert response.status_code == 200
        
        # Invalid k value
        response = self.client.post("/memories/search", json={
            "query": "test",
            "k": 0  # Below minimum
        })
        assert response.status_code == 422
        
        response = self.client.post("/memories/search", json={
            "query": "test",
            "k": 100  # Above maximum
        })
        assert response.status_code == 422
        
        # Empty query
        response = self.client.post("/memories/search", json={
            "query": ""
        })
        assert response.status_code == 422
    
    def test_memory_request_validation(self):
        """Test MemoryRequest model validation and enhanced fields"""
        # Valid request with all fields
        valid_request = {
            "content": "Test memory content",
            "metadata": {"key": "value"},
            "role": "user",
            "type": "history", 
            "importance": 0.8
        }
        
        response = self.client.post("/memories", json=valid_request)
        assert response.status_code == 200
        
        # Check that add_memory was called with enhanced fields
        mock_memory_engine.add_memory.assert_called_once()
        call_args = mock_memory_engine.add_memory.call_args
        
        assert call_args.kwargs["content"] == "Test memory content"
        assert call_args.kwargs["metadata"] == {"key": "value"}
        assert call_args.kwargs["role"] == "user"
        assert call_args.kwargs["type"] == "history"
        assert call_args.kwargs["importance"] == 0.8
        
        # Test validation failures
        response = self.client.post("/memories", json={
            "content": "",  # Empty content
        })
        assert response.status_code == 422
        
        response = self.client.post("/memories", json={
            "content": "test",
            "importance": 2.0  # Above maximum
        })
        assert response.status_code == 422
    
    def test_ingest_endpoint_background_processing(self):
        """Test ingest endpoint with background task processing"""
        request_data = {
            "content": "Content to ingest",
            "source": "test_source",
            "metadata": {"type": "test"}
        }
        
        response = self.client.post("/ingest", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "queued"
        assert data["content_length"] == len(request_data["content"])
        assert data["source"] == "test_source"
    
    def test_github_webhook_hmac_verification(self):
        """Test GitHub webhook HMAC signature verification"""
        # Mock the SECRET environment variable
        test_secret = "test_webhook_secret"
        
        with patch('chatgpt_memory_api.SECRET', test_secret):
            payload = {"action": "push", "repository": {"name": "test-repo"}}
            payload_bytes = json.dumps(payload).encode()
            
            # Generate valid signature
            mac = hmac.new(test_secret.encode(), msg=payload_bytes, digestmod=hashlib.sha256)
            valid_signature = f"sha256={mac.hexdigest()}"
            
            # Test with valid signature
            response = self.client.post(
                "/webhook/github",
                content=payload_bytes,
                headers={
                    "X-Hub-Signature-256": valid_signature,
                    "X-GitHub-Event": "push",
                    "Content-Type": "application/json"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
            assert data["event_type"] == "push"
    
    def test_github_webhook_invalid_signature(self):
        """Test GitHub webhook with invalid signature"""
        test_secret = "test_webhook_secret"
        
        with patch('chatgpt_memory_api.SECRET', test_secret):
            payload = {"action": "push"}
            payload_bytes = json.dumps(payload).encode()
            
            # Invalid signature
            response = self.client.post(
                "/webhook/github",
                content=payload_bytes,
                headers={
                    "X-Hub-Signature-256": "sha256=invalid_signature",
                    "X-GitHub-Event": "push"
                }
            )
            
            assert response.status_code == 401
            assert "Invalid signature" in response.json()["detail"]
    
    def test_github_webhook_no_secret_configured(self):
        """Test GitHub webhook when no secret is configured"""
        with patch('chatgpt_memory_api.SECRET', ''):
            payload = {"action": "push"}
            payload_bytes = json.dumps(payload).encode()
            
            # Should succeed without signature verification
            response = self.client.post(
                "/webhook/github",
                content=payload_bytes,
                headers={"X-GitHub-Event": "push"}
            )
            
            assert response.status_code == 200
    
    def test_cleanup_endpoint(self):
        """Test memory cleanup endpoint"""
        with patch('core.memory_manager.create_default_memory_manager') as mock_manager_factory:
            mock_manager = Mock()
            mock_stats = Mock()
            mock_stats.total_memories_before = 1000
            mock_stats.total_memories_after = 800
            mock_stats.memories_cleaned = 200
            mock_stats.memories_archived = 150
            mock_stats.duration_ms = 5000
            
            mock_manager.auto_cleanup.return_value = mock_stats
            mock_manager_factory.return_value = mock_manager
            
            response = self.client.post("/memories/cleanup", json={
                "max_memories": 800,
                "max_age_days": 30,
                "dry_run": False
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["memories_before"] == 1000
            assert data["memories_after"] == 800
            assert data["memories_cleaned"] == 200
            assert data["memories_archived"] == 150
    
    def test_cleanup_endpoint_validation(self):
        """Test cleanup endpoint validation"""
        # No criteria specified
        response = self.client.post("/memories/cleanup", json={
            "dry_run": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "No cleanup criteria specified" in data["error"]
        
        # Invalid max_memories
        response = self.client.post("/memories/cleanup", json={
            "max_memories": 50  # Below minimum
        })
        assert response.status_code == 422
    
    def test_metrics_endpoint_prometheus_available(self):
        """Test metrics endpoint when Prometheus is available"""
        with patch('prometheus_client.generate_latest') as mock_generate:
            with patch('prometheus_client.CONTENT_TYPE_LATEST', 'text/plain'):
                mock_generate.return_value = b"# HELP test_metric Test metric\n"
                
                response = self.client.get("/metrics")
                
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/plain")
    
    def test_metrics_endpoint_prometheus_unavailable(self):
        """Test metrics endpoint when Prometheus is not available"""
        with patch('chatgpt_memory_api.prometheus_client', side_effect=ImportError):
            response = self.client.get("/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert "not available" in data["message"]
    
    def test_request_response_consistency(self):
        """Test that request/response models are consistent"""
        # Test that chat endpoint properly uses ChatRequest fields
        with patch('chatgpt_memory_api.generate_gpt_response') as mock_gpt:
            mock_gpt.return_value = "Test response"
            
            response = self.client.post("/chat", json={
                "message": "test message",
                "use_gpt": False,  # Should be respected
                "context_k": 8     # Should be used for search
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Should indicate no GPT was used
            assert data["response_type"] in ["context-only", "simple"]
    
    def test_error_handling_consistency(self):
        """Test consistent error handling across endpoints"""
        # Test invalid JSON
        response = self.client.post(
            "/memories",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test missing required field
        response = self.client.post("/memories", json={})
        assert response.status_code == 422
        
        # Test field validation
        response = self.client.post("/memories", json={
            "content": "x" * 50001  # Too long
        })
        assert response.status_code == 422
    
    def test_content_type_handling(self):
        """Test proper content type handling"""
        # JSON content should work
        response = self.client.post("/memories", json={"content": "test"})
        assert response.status_code == 200
        
        # Form data should fail for JSON endpoints
        response = self.client.post("/memories", data={"content": "test"})
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])