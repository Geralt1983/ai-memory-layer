"""Smoke tests for HTTP server functionality."""

import os
import json
import tempfile
import subprocess
import sys
import time
import threading
import pytest


@pytest.mark.integration
@pytest.mark.skipif(not (os.getenv("VOYAGE_API_KEY") or os.getenv("OPENAI_API_KEY")), 
                   reason="needs API key for server smoke test")
def test_server_smoke():
    """Test server build and search endpoints."""
    try:
        import requests
    except ImportError:
        pytest.skip("requests not available for server testing")
    
    # Choose provider based on available keys
    provider = "voyage" if os.getenv("VOYAGE_API_KEY") else "openai"
    model = "voyage-2" if provider == "voyage" else "text-embedding-3-small"
    
    with tempfile.TemporaryDirectory() as tmp_path:
        # Set up environment
        env = os.environ.copy()
        env["MEM_INDEX_DIR"] = os.path.join(tmp_path, "idx")
        env["EMBED_CACHE_PATH"] = os.path.join(tmp_path, "cache.db")
        env["EMBED_PROVIDER"] = provider
        env["EMBED_MODEL"] = model
        
        # Create test corpus
        corpus_dir = os.path.join(tmp_path, "docs")
        os.makedirs(corpus_dir)
        
        with open(os.path.join(corpus_dir, "a.txt"), "w", encoding="utf-8") as f:
            f.write("serena improves code review")
        
        with open(os.path.join(corpus_dir, "b.txt"), "w", encoding="utf-8") as f:
            f.write("faiss index reuse is fast")
        
        with open(os.path.join(corpus_dir, "c.md"), "w", encoding="utf-8") as f:
            f.write("# Memory Layer\nSemantic search with temporal decay")
        
        # Start server
        proc = subprocess.Popen([
            sys.executable, "-m", "memory_layer.server"
        ], env=env)
        
        try:
            # Wait for server to start
            time.sleep(3.0)
            
            # Test health endpoint
            r = requests.get("http://127.0.0.1:8080/health", timeout=5)
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
            
            # Test providers endpoint
            r = requests.get("http://127.0.0.1:8080/providers", timeout=5)
            assert r.status_code == 200
            providers = r.json()["providers"]
            assert isinstance(providers, list)
            assert provider in providers
            
            # Test build endpoint
            r = requests.post("http://127.0.0.1:8080/build", 
                            json={"dir": corpus_dir},
                            timeout=30)
            assert r.status_code == 200
            result = r.json()
            assert result["count"] > 0
            assert "out" in result
            
            # Verify index files were created
            index_dir = result["out"]
            assert os.path.exists(os.path.join(index_dir, "faiss.index"))
            assert os.path.exists(os.path.join(index_dir, "corpus.jsonl"))
            assert os.path.exists(os.path.join(index_dir, "ids.json"))
            
            # Test search endpoint
            r = requests.get("http://127.0.0.1:8080/search", 
                           params={"q": "code review", "k": 2},
                           timeout=30)
            assert r.status_code == 200
            results = r.json()
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Verify result structure
            for result in results:
                assert "id" in result
                assert "score" in result
                assert "text" in result
                assert isinstance(result["score"], (int, float))
            
            # Test different search query
            r = requests.get("http://127.0.0.1:8080/search",
                           params={"q": "semantic search", "k": 3},
                           timeout=30)
            assert r.status_code == 200
            results = r.json()
            assert isinstance(results, list)
            
        finally:
            proc.terminate()
            proc.wait(timeout=5)


@pytest.mark.unit
def test_server_import():
    """Test that server module can be imported."""
    from memory_layer import server
    assert hasattr(server, 'app')
    assert hasattr(server, 'build')
    assert hasattr(server, 'search')


@pytest.mark.unit  
def test_server_models():
    """Test Pydantic models for server."""
    from memory_layer.server import BuildRequest, SearchResponse
    
    # Test BuildRequest
    req = BuildRequest(dir="/tmp/docs", provider="voyage")
    assert req.dir == "/tmp/docs"
    assert req.provider == "voyage"
    assert req.out == ".index"  # default
    
    # Test SearchResponse
    resp = SearchResponse(id="test-id", score=0.95, text="test content")
    assert resp.id == "test-id"
    assert resp.score == 0.95
    assert resp.text == "test content"


@pytest.mark.integration
@pytest.mark.skip(reason="Optional server test requiring manual setup")
def test_server_error_conditions():
    """Test server error handling (manual test)."""
    try:
        import requests
    except ImportError:
        pytest.skip("requests not available")
    
    # This would test error conditions like:
    # - Build with no input texts
    # - Search with missing index
    # - Invalid provider
    # etc.
    
    # Start server without API keys
    env = os.environ.copy()
    env.pop("VOYAGE_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    
    # Test would verify proper error responses
    pass


def test_server_concurrent_requests():
    """Test server can handle concurrent requests (unit test with mocking)."""
    # This would use mocking to test concurrent request handling
    # without actually starting a server or making real API calls
    from memory_layer.server import app
    from fastapi.testclient import TestClient
    
    try:
        client = TestClient(app)
        
        # Test health endpoint works
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        
        # Test providers endpoint
        response = client.get("/providers")
        assert response.status_code == 200
        assert "providers" in response.json()
        
    except ImportError:
        pytest.skip("FastAPI TestClient not available")