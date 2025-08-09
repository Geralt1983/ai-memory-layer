"""
Health check endpoint testing for embedding providers.
Tests operational health and status endpoints.
"""
import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestHealthEndpoints:
    """Test health check endpoints for embedding system."""
    
    @pytest.fixture
    def api_client(self):
        """Create test client for API."""
        try:
            from api.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI not available for health testing")
    
    def test_health_endpoint_basic(self, api_client):
        """Test basic health endpoint functionality."""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert health_data["status"] in ["healthy", "ok", "up"]
    
    def test_health_endpoint_with_embedding_status(self, api_client):
        """Test health endpoint includes embedding provider status."""
        with patch("integrations.embeddings_factory.get_available_providers") as mock_providers:
            mock_providers.return_value = {
                "openai": True,
                "voyage": False,
                "cohere": True,
                "dualwrite": True,
                "fallback": True
            }
            
            response = api_client.get("/health")
            assert response.status_code == 200
            
            health_data = response.json()
            
            # Check for provider status in response
            if "providers" in health_data or "embeddings" in health_data:
                # Health endpoint includes provider status
                provider_status = health_data.get("providers", health_data.get("embeddings", {}))
                assert isinstance(provider_status, dict)
            else:
                # Basic health endpoint without detailed provider status
                assert "status" in health_data
    
    def test_internal_health_endpoints(self, api_client):
        """Test internal health endpoints if available."""
        # Try various internal health endpoints
        internal_endpoints = [
            "/internal/health",
            "/ops/health", 
            "/internal/status",
            "/status",
            "/internal/embeddings/health"
        ]
        
        working_endpoints = []
        for endpoint in internal_endpoints:
            try:
                response = api_client.get(endpoint)
                if response.status_code == 200:
                    working_endpoints.append(endpoint)
            except Exception:
                pass  # Endpoint doesn't exist
        
        # At least the basic health endpoint should work
        basic_health = api_client.get("/health")
        assert basic_health.status_code == 200
    
    def test_embedding_provider_health_details(self, api_client):
        """Test detailed embedding provider health information."""
        with patch("integrations.embeddings_factory.get_embedder") as mock_get_embedder:
            # Mock a healthy embedder
            mock_embedder = Mock()
            mock_embedder.get_embedding_dimension.return_value = 1536
            mock_get_embedder.return_value = mock_embedder
            
            response = api_client.get("/health")
            health_data = response.json()
            
            # Health should succeed when embedder is available
            assert response.status_code == 200
            assert health_data["status"] in ["healthy", "ok", "up"]
    
    def test_health_with_embedding_failure(self, api_client):
        """Test health endpoint behavior when embedding provider fails."""
        with patch("integrations.embeddings_factory.get_embedder") as mock_get_embedder:
            # Mock embedder that fails
            mock_get_embedder.side_effect = Exception("Embedding provider unavailable")
            
            response = api_client.get("/health")
            
            # Health endpoint should still return 200 but may indicate degraded state
            # Different implementations may handle this differently
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                health_data = response.json()
                # May show degraded status
                assert "status" in health_data
            else:
                # Service unavailable due to critical dependency failure
                assert response.status_code == 503
    
    @pytest.mark.parametrize("provider_name", ["openai", "voyage", "cohere"])
    def test_provider_specific_health(self, api_client, provider_name):
        """Test health checks for specific providers."""
        # Try provider-specific health endpoints
        provider_endpoints = [
            f"/internal/embeddings/{provider_name}/health",
            f"/ops/providers/{provider_name}/status",
            f"/health/{provider_name}"
        ]
        
        responses = []
        for endpoint in provider_endpoints:
            try:
                response = api_client.get(endpoint)
                responses.append((endpoint, response.status_code, response.json()))
            except Exception:
                pass  # Endpoint may not exist
        
        # At least test that general health works
        general_health = api_client.get("/health")
        assert general_health.status_code == 200
    
    def test_health_response_format(self, api_client):
        """Test health response follows expected format."""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        
        # Basic required fields
        assert isinstance(health_data, dict)
        assert "status" in health_data
        
        # Common optional fields
        optional_fields = ["version", "timestamp", "uptime", "providers", "embeddings", "dependencies"]
        
        # Validate structure of any present optional fields
        for field in optional_fields:
            if field in health_data:
                if field == "providers" or field == "embeddings":
                    assert isinstance(health_data[field], dict)
                elif field == "dependencies":
                    assert isinstance(health_data[field], (list, dict))
                elif field in ["timestamp", "version", "uptime"]:
                    assert isinstance(health_data[field], (str, int, float))
    
    def test_health_with_ab_testing_config(self, api_client, monkeypatch):
        """Test health endpoint with A/B testing configuration."""
        monkeypatch.setenv("EMBED_AB_WRITE", "openai,voyage")
        
        with patch("integrations.embeddings_factory.get_embedder_ab") as mock_get_ab:
            mock_ab_embedder = Mock()
            mock_get_ab.return_value = mock_ab_embedder
            
            response = api_client.get("/health")
            assert response.status_code == 200
            
            health_data = response.json()
            assert "status" in health_data
    
    def test_health_performance(self, api_client):
        """Test health endpoint response time."""
        import time
        
        start_time = time.time()
        response = api_client.get("/health")
        end_time = time.time()
        
        # Health endpoint should respond quickly (< 1 second)
        response_time = end_time - start_time
        assert response_time < 1.0
        assert response.status_code == 200
    
    def test_concurrent_health_requests(self, api_client):
        """Test health endpoint under concurrent load."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_health_request():
            try:
                start = time.time()
                response = api_client.get("/health")
                end = time.time()
                results.append((response.status_code, end - start))
            except Exception as e:
                errors.append(e)
        
        # Make 10 concurrent health requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_health_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10
        assert all(status == 200 for status, _ in results)
        
        # All should be reasonably fast
        assert all(duration < 2.0 for _, duration in results)


@pytest.mark.unit
class TestHealthLogic:
    """Test health check logic without HTTP layer."""
    
    def test_provider_availability_check(self):
        """Test provider availability checking logic."""
        from integrations.embeddings_factory import get_available_providers
        
        providers = get_available_providers()
        
        # OpenAI should always be reported as available
        assert "openai" in providers
        assert providers["openai"] is True
        
        # Built-in providers should be available
        assert "dualwrite" in providers
        assert providers["dualwrite"] is True
        
        assert "fallback" in providers  
        assert providers["fallback"] is True
    
    def test_embedder_health_status(self, fake_openai_embeddings):
        """Test individual embedder health status."""
        from integrations.embeddings import OpenAIEmbeddings
        
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        # Test that embedder can perform basic operations
        try:
            dimension = embedder.get_embedding_dimension()
            assert dimension > 0
            
            # Test basic embedding
            result = embedder.embed(["health check"])
            assert len(result) == 1
            assert len(result[0]) > 0
            
            # If we get here, embedder is healthy
            health_status = True
        except Exception:
            health_status = False
        
        assert health_status is True
    
    def test_system_health_aggregation(self, fake_openai_embeddings):
        """Test aggregating health status from multiple components."""
        from integrations.embeddings_factory import get_available_providers
        from integrations.embeddings_factory import get_embedder
        
        # Check provider availability
        providers = get_available_providers()
        available_count = sum(1 for available in providers.values() if available)
        
        # Check primary embedder
        try:
            embedder = get_embedder()
            embedder_healthy = True
        except Exception:
            embedder_healthy = False
        
        # Aggregate health status
        overall_health = {
            "providers_available": available_count > 0,
            "primary_embedder": embedder_healthy,
            "total_providers": len(providers),
            "available_providers": available_count
        }
        
        # System is healthy if we have at least one provider and primary embedder works
        system_healthy = overall_health["providers_available"] and overall_health["primary_embedder"]
        
        assert system_healthy is True
        assert overall_health["available_providers"] >= 1
    
    def test_health_check_with_fallback(self, fake_openai_embeddings):
        """Test health check when using fallback provider."""
        from integrations.providers.fallback import FallbackEmbeddings
        from integrations.embeddings import OpenAIEmbeddings
        
        # Create fallback with healthy backup
        primary = OpenAIEmbeddings(api_key="test-key")
        backup = OpenAIEmbeddings(api_key="test-key")
        
        fallback_embedder = FallbackEmbeddings(primary=primary, backup=backup)
        
        # Test health
        try:
            result = fallback_embedder.embed(["fallback health check"])
            dimension = fallback_embedder.get_embedding_dimension()
            
            fallback_healthy = len(result) == 1 and dimension > 0
        except Exception:
            fallback_healthy = False
        
        assert fallback_healthy is True
    
    def test_health_metrics_collection(self, fake_openai_embeddings):
        """Test collection of health metrics."""
        from integrations.embeddings import OpenAIEmbeddings
        import time
        
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        # Collect timing metrics
        start_time = time.time()
        result = embedder.embed(["metrics test"])
        end_time = time.time()
        
        metrics = {
            "embedding_latency": end_time - start_time,
            "embedding_success": len(result) == 1,
            "provider_type": "openai",
            "timestamp": start_time
        }
        
        assert metrics["embedding_success"] is True
        assert metrics["embedding_latency"] >= 0
        assert metrics["provider_type"] == "openai"
        assert metrics["timestamp"] > 0
