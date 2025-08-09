"""Tests for embedding provider routing functionality."""

import os
import pytest
from unittest.mock import Mock, patch
from typing import List

from integrations.embeddings_factory import get_embedder_for, _parse_routing


def test_parse_routing_basic():
    """Test basic routing string parsing."""
    routing = _parse_routing("obsidian:openai,commits:voyage,default:fallback")
    expected = {
        "obsidian": "openai",
        "commits": "voyage", 
        "default": "fallback"
    }
    assert routing == expected


def test_parse_routing_empty():
    """Test parsing empty routing string."""
    assert _parse_routing("") == {}
    assert _parse_routing("   ") == {}


def test_parse_routing_malformed():
    """Test parsing malformed routing strings."""
    # Missing colons should be ignored
    routing = _parse_routing("obsidian:openai,bad_entry,commits:voyage")
    expected = {
        "obsidian": "openai",
        "commits": "voyage"
    }
    assert routing == expected
    
    # Extra colons - should take first split
    routing = _parse_routing("tag:provider:extra:parts")
    expected = {"tag": "provider:extra:parts"}
    assert routing == expected


def test_parse_routing_whitespace():
    """Test parsing with whitespace."""
    routing = _parse_routing("  obsidian : openai , commits: voyage  ")
    expected = {
        "obsidian": "openai",
        "commits": "voyage"
    }
    assert routing == expected


class MockProvider:
    """Mock embedding provider for testing."""
    def __init__(self, name: str):
        self.name = name
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Return deterministic vectors based on provider name
        return [[float(ord(self.name[0]))] * 10 for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        return 10


@pytest.fixture
def mock_providers():
    """Mock provider creation functions."""
    def mock_openai():
        return MockProvider("openai")
    
    def mock_voyage():
        return MockProvider("voyage")
    
    def mock_fallback():
        return MockProvider("fallback")
    
    return {
        "openai": mock_openai,
        "voyage": mock_voyage,
        "fallback": mock_fallback
    }


def test_get_embedder_for_no_routing(mock_providers, monkeypatch):
    """Test get_embedder_for with no routing configured."""
    # Mock the standard get_embedder to return openai
    with patch('integrations.embeddings_factory.get_embedder') as mock_get:
        mock_get.return_value = MockProvider("openai")
        
        # No EMBED_ROUTING env var
        monkeypatch.delenv("EMBED_ROUTING", raising=False)
        
        embedder = get_embedder_for("obsidian")
        assert embedder.name == "openai"
        mock_get.assert_called_once()


def test_get_embedder_for_with_routing(mock_providers, monkeypatch):
    """Test get_embedder_for with routing configured."""
    # Set up routing
    monkeypatch.setenv("EMBED_ROUTING", "obsidian:openai,commits:voyage,default:fallback")
    
    # Mock _create_provider
    def mock_create_provider(provider_name):
        return mock_providers[provider_name]()
    
    with patch('integrations.embeddings_factory._create_provider', side_effect=mock_create_provider):
        # Test specific routes
        obsidian_embedder = get_embedder_for("obsidian")
        assert obsidian_embedder.name == "openai"
        
        commits_embedder = get_embedder_for("commits")
        assert commits_embedder.name == "voyage"
        
        # Test default route
        unknown_embedder = get_embedder_for("unknown_tag")
        assert unknown_embedder.name == "fallback"
        
        # Test case insensitive
        caps_embedder = get_embedder_for("OBSIDIAN")
        assert caps_embedder.name == "openai"


def test_get_embedder_for_no_default(mock_providers, monkeypatch):
    """Test get_embedder_for when no default route and tag not found."""
    # Set up routing without default
    monkeypatch.setenv("EMBED_ROUTING", "obsidian:openai,commits:voyage")
    
    with patch('integrations.embeddings_factory.get_embedder') as mock_get:
        mock_get.return_value = MockProvider("standard")
        
        # Unknown tag with no default should fall back to standard get_embedder
        embedder = get_embedder_for("unknown")
        assert embedder.name == "standard"
        mock_get.assert_called_once()


def test_routing_integration_vectors(mock_providers, monkeypatch):
    """Test that routing produces different vectors for different providers."""
    monkeypatch.setenv("EMBED_ROUTING", "typeA:openai,typeB:voyage")
    
    def mock_create_provider(provider_name):
        return mock_providers[provider_name]()
    
    with patch('integrations.embeddings_factory._create_provider', side_effect=mock_create_provider):
        embedder_a = get_embedder_for("typeA")
        embedder_b = get_embedder_for("typeB")
        
        vectors_a = embedder_a.embed(["test"])
        vectors_b = embedder_b.embed(["test"])
        
        # Should produce different vectors (different first elements)
        assert vectors_a[0][0] != vectors_b[0][0]
        assert vectors_a[0][0] == float(ord("o"))  # openai -> 'o' -> 111.0
        assert vectors_b[0][0] == float(ord("v"))  # voyage -> 'v' -> 118.0