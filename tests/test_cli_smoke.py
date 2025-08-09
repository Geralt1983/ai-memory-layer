"""CLI smoke tests for build and search functionality."""

import os
import json
import tempfile
import subprocess
import sys
import pytest


@pytest.mark.integration
def test_cli_build_and_search():
    """Test CLI build and search with real providers (if keys available)."""
    # Skip if no provider keys available
    has_voyage = bool(os.getenv("VOYAGE_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not (has_voyage or has_openai):
        pytest.skip("No API keys available (VOYAGE_API_KEY or OPENAI_API_KEY)")
    
    # Choose provider based on available keys
    if has_voyage:
        provider = "voyage"
        model = os.getenv("EMBED_MODEL", "voyage-2")
    else:
        provider = "openai"
        model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    
    dim = int(os.getenv("EMBED_DIM", "1536"))
    
    with tempfile.TemporaryDirectory() as tmp_path:
        # Prepare tiny corpus
        corpus_dir = os.path.join(tmp_path, "docs")
        os.makedirs(corpus_dir)
        
        with open(os.path.join(corpus_dir, "a.txt"), "w", encoding="utf-8") as f:
            f.write("the quick brown fox")
        
        with open(os.path.join(corpus_dir, "b.txt"), "w", encoding="utf-8") as f:
            f.write("jumps over the lazy dog")
        
        with open(os.path.join(corpus_dir, "c.md"), "w", encoding="utf-8") as f:
            f.write("# AI Memory Layer\n\nThis is a test document about embeddings")
        
        outdir = os.path.join(tmp_path, "idx")
        
        # Build index
        cmd_build = [
            sys.executable, "-m", "memory_layer.cli", "build",
            "--dir", corpus_dir,
            "--out", outdir,
            "--provider", provider,
            "--model", model,
            "--dim", str(dim),
            "--cache", os.path.join(tmp_path, "cache.db")
        ]
        
        result = subprocess.run(cmd_build, capture_output=True, text=True)
        print(f"Build stdout: {result.stdout}")
        print(f"Build stderr: {result.stderr}")
        assert result.returncode == 0, f"Build failed: {result.stderr}"
        
        # Check that index files were created
        assert os.path.exists(os.path.join(outdir, "faiss.index"))
        assert os.path.exists(os.path.join(outdir, "faiss.index.meta.json"))
        
        # Search index
        cmd_search = [
            sys.executable, "-m", "memory_layer.cli", "search",
            "--out", outdir,
            "--provider", provider,
            "--model", model,
            "--dim", str(dim),
            "--cache", os.path.join(tmp_path, "cache.db"),
            "brown fox"
        ]
        
        result = subprocess.run(cmd_search, capture_output=True, text=True)
        print(f"Search stdout: {result.stdout}")
        print(f"Search stderr: {result.stderr}")
        assert result.returncode == 0, f"Search failed: {result.stderr}"
        
        # Parse search results
        parsed = json.loads(result.stdout)
        assert "scores" in parsed
        assert "ids" in parsed
        assert len(parsed["scores"]) > 0
        assert len(parsed["ids"]) > 0
        assert len(parsed["scores"]) == len(parsed["ids"])


@pytest.mark.unit
def test_cli_build_no_input():
    """Test that CLI build fails gracefully with no input."""
    with tempfile.TemporaryDirectory() as tmp_path:
        empty_dir = os.path.join(tmp_path, "empty")
        os.makedirs(empty_dir)
        
        cmd_build = [
            sys.executable, "-m", "memory_layer.cli", "build",
            "--dir", empty_dir,
            "--out", os.path.join(tmp_path, "idx"),
            "--provider", "voyage"  # Won't matter since no input
        ]
        
        result = subprocess.run(cmd_build, capture_output=True, text=True)
        assert result.returncode == 1  # Should fail with no input
        assert "no input texts" in result.stderr


@pytest.mark.unit
def test_cli_search_no_index():
    """Test that CLI search fails gracefully when index doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp_path:
        nonexistent_dir = os.path.join(tmp_path, "nonexistent")
        
        cmd_search = [
            sys.executable, "-m", "memory_layer.cli", "search",
            "--out", nonexistent_dir,
            "--provider", "voyage",
            "test query"
        ]
        
        result = subprocess.run(cmd_search, capture_output=True, text=True)
        assert result.returncode == 1
        assert "index not found" in result.stderr


@pytest.mark.unit
def test_cli_unavailable_provider():
    """Test CLI behavior when provider is not available."""
    # Clear environment to ensure no API keys
    env = os.environ.copy()
    env.pop("VOYAGE_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    
    with tempfile.TemporaryDirectory() as tmp_path:
        corpus_dir = os.path.join(tmp_path, "docs")
        os.makedirs(corpus_dir)
        
        with open(os.path.join(corpus_dir, "test.txt"), "w") as f:
            f.write("test content")
        
        cmd_build = [
            sys.executable, "-m", "memory_layer.cli", "build",
            "--dir", corpus_dir,
            "--out", os.path.join(tmp_path, "idx"),
            "--provider", "voyage"
        ]
        
        result = subprocess.run(cmd_build, capture_output=True, text=True, env=env)
        assert result.returncode == 2  # Provider unavailable
        assert "not available" in result.stderr


@pytest.mark.integration 
@pytest.mark.skipif(not (os.getenv("VOYAGE_API_KEY") or os.getenv("OPENAI_API_KEY")), 
                   reason="needs API key for integration test")
def test_cli_jsonl_input():
    """Test CLI with JSONL input format."""
    # Choose available provider
    provider = "voyage" if os.getenv("VOYAGE_API_KEY") else "openai"
    model = "voyage-2" if provider == "voyage" else "text-embedding-3-small"
    
    with tempfile.TemporaryDirectory() as tmp_path:
        # Create JSONL file
        jsonl_path = os.path.join(tmp_path, "data.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write('{"text": "first document about cats"}\n')
            f.write('{"text": "second document about dogs"}\n')
            f.write('{"text": "third document about birds"}\n')
        
        outdir = os.path.join(tmp_path, "idx")
        
        # Build from JSONL
        cmd_build = [
            sys.executable, "-m", "memory_layer.cli", "build",
            "--jsonl", jsonl_path,
            "--out", outdir,
            "--provider", provider,
            "--model", model
        ]
        
        result = subprocess.run(cmd_build, capture_output=True, text=True)
        assert result.returncode == 0, f"Build failed: {result.stderr}"
        
        # Search should work
        cmd_search = [
            sys.executable, "-m", "memory_layer.cli", "search",
            "--out", outdir,
            "--provider", provider,
            "--model", model,
            "cats and dogs"
        ]
        
        result = subprocess.run(cmd_search, capture_output=True, text=True)
        assert result.returncode == 0, f"Search failed: {result.stderr}"
        
        parsed = json.loads(result.stdout)
        assert len(parsed["scores"]) > 0


@pytest.mark.unit
def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run([
        sys.executable, "-m", "memory_layer.cli", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "mem-index" in result.stdout
    assert "build" in result.stdout
    assert "search" in result.stdout


@pytest.mark.unit
def test_cli_build_help():
    """Test that CLI build help works."""
    result = subprocess.run([
        sys.executable, "-m", "memory_layer.cli", "build", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "--dir" in result.stdout
    assert "--jsonl" in result.stdout
    assert "--provider" in result.stdout


@pytest.mark.unit
def test_cli_search_help():
    """Test that CLI search help works."""
    result = subprocess.run([
        sys.executable, "-m", "memory_layer.cli", "search", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "query" in result.stdout
    assert "--provider" in result.stdout
    assert "-k" in result.stdout