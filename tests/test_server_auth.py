"""Tests for HTTP server authentication and security."""

import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pathlib import Path

from memory_layer.server import app, INDEX_DIR, _require_api_key, _safe_out


@pytest.fixture
def client():
    """Test client without authentication."""
    return TestClient(app)


@pytest.fixture
def auth_client():
    """Test client with API key authentication enabled."""
    with patch.dict(os.environ, {"SERVER_API_KEY": "test-key-123"}):
        return TestClient(app)


@pytest.mark.unit
class TestServerSecurity:
    """Test HTTP server security features."""
    
    def test_health_endpoint_no_auth_required(self, client):
        """Test that health endpoint works without auth."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_providers_endpoint_no_auth_required(self, client):
        """Test that providers endpoint works without auth."""
        response = client.get("/providers")
        assert response.status_code == 200
        assert "providers" in response.json()
    
    def test_search_no_auth_required_when_disabled(self, client):
        """Test that search works when auth is disabled."""
        # This will fail due to missing index, but should not fail auth
        response = client.get("/search?q=test")
        assert response.status_code != 403  # Should not be auth error
    
    def test_build_requires_auth_when_enabled(self, auth_client):
        """Test that build endpoint requires auth when enabled."""
        response = auth_client.post("/build", json={"dir": "/tmp"})
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]
    
    def test_build_works_with_valid_auth(self, auth_client):
        """Test that build works with valid API key."""
        headers = {"x-api-key": "test-key-123"}
        # Will fail due to missing texts, but should pass auth
        response = auth_client.post("/build", json={"dir": "/nonexistent"}, headers=headers)
        assert response.status_code != 403  # Should not be auth error
    
    def test_require_api_key_function(self):
        """Test _require_api_key function directly."""
        from fastapi import Request, HTTPException
        
        # Mock request without API key
        request = MagicMock(spec=Request)
        request.headers.get.return_value = None
        
        with patch.dict(os.environ, {"SERVER_API_KEY": "secret"}):
            with pytest.raises(HTTPException) as exc_info:
                _require_api_key(request)
            assert exc_info.value.status_code == 403
            assert "Invalid API key" in str(exc_info.value.detail)
    
    def test_require_api_key_disabled(self):
        """Test that auth is skipped when SERVER_API_KEY is empty."""
        from fastapi import Request
        
        request = MagicMock(spec=Request)
        
        with patch.dict(os.environ, {"SERVER_API_KEY": ""}):
            # Should not raise
            result = _require_api_key(request)
            assert result is None


@pytest.mark.unit
class TestPathTraversal:
    """Test path traversal protection."""
    
    def test_safe_out_within_index_dir(self):
        """Test that paths within INDEX_DIR are allowed."""
        test_path = "subdir/test"
        result = _safe_out(test_path)
        expected = INDEX_DIR / test_path
        assert str(result).startswith(str(INDEX_DIR))
    
    def test_safe_out_blocks_traversal(self):
        """Test that path traversal attempts are blocked."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            _safe_out("../../../etc/passwd")
        assert exc_info.value.status_code == 400
        assert "must be under MEM_INDEX_DIR" in str(exc_info.value.detail)
    
    def test_safe_out_blocks_absolute_paths_outside(self):
        """Test that absolute paths outside INDEX_DIR are blocked."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            _safe_out("/tmp/malicious")
        assert exc_info.value.status_code == 400
    
    def test_safe_out_creates_directory(self):
        """Test that _safe_out creates directories."""
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('memory_layer.server.INDEX_DIR', Path(tmpdir)):
                test_path = "new/nested/dir"
                result = _safe_out(test_path)
                assert result.exists()
                assert result.is_dir()
    
    def test_safe_out_symlink_protection(self):
        """Test that symlinks cannot escape INDEX_DIR."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('memory_layer.server.INDEX_DIR', Path(tmpdir)):
                # Create a symlink that tries to escape
                escape_target = Path(tmpdir).parent / "escaped"
                escape_target.mkdir(exist_ok=True)
                
                symlink_path = Path(tmpdir) / "escape_link"
                try:
                    symlink_path.symlink_to(escape_target)
                    
                    # Should be blocked when resolved
                    from fastapi import HTTPException
                    with pytest.raises(HTTPException):
                        _safe_out(str(symlink_path.relative_to(Path(tmpdir))))
                except OSError:
                    # Skip if symlinks not supported on this system
                    pytest.skip("Symlinks not supported")


@pytest.mark.integration
class TestServerEndToEnd:
    """End-to-end server tests with security."""
    
    def test_build_endpoint_security_flow(self):
        """Test complete build endpoint security flow."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test text file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content for embedding")
            
            with patch.dict(os.environ, {"SERVER_API_KEY": "secure-key-456"}):
                client = TestClient(app)
                
                # Without auth header should fail
                response = client.post("/build", json={"dir": tmpdir})
                assert response.status_code == 403
                
                # With wrong auth should fail  
                response = client.post("/build", 
                                     json={"dir": tmpdir},
                                     headers={"x-api-key": "wrong-key"})
                assert response.status_code == 403
                
                # With correct auth should pass auth check
                # (may fail later due to missing provider config)
                response = client.post("/build",
                                     json={"dir": tmpdir},
                                     headers={"x-api-key": "secure-key-456"})
                assert response.status_code != 403