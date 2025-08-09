"""Internal endpoints for operational monitoring and debugging."""

import os
from fastapi import APIRouter, HTTPException
from integrations.embeddings_factory import get_embedder, get_available_providers, get_embedder_for

router = APIRouter(prefix="/internal", tags=["internal"])


@router.get("/embeddings/health")
async def embeddings_health():
    """Health check for the active embedding provider.
    
    Returns the active provider and tests a basic embedding operation.
    """
    provider_name = os.getenv("EMBEDDING_PROVIDER", "openai")
    
    try:
        embedder = get_embedder()
        # Test with a simple embedding to verify it's working
        test_result = embedder.embed(["health check"])
        
        return {
            "provider": provider_name,
            "ok": True,
            "test_embedding_count": len(test_result),
            "embedding_dimension": embedder.get_embedding_dimension() if hasattr(embedder, 'get_embedding_dimension') else None
        }
    except Exception as e:
        return {
            "provider": provider_name,
            "ok": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@router.get("/embeddings/providers")
async def embeddings_providers():
    """List all available embedding providers and their status."""
    return {
        "active_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
        "available_providers": get_available_providers(),
        "config": {
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
            "openai_embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002"),
            "embedding_cache_enabled": os.getenv("EMBEDDING_CACHE_ENABLED", "false"),
            "embedding_primary_provider": os.getenv("EMBEDDING_PRIMARY_PROVIDER"),
            "embedding_fallback_provider": os.getenv("EMBEDDING_FALLBACK_PROVIDER"),
            "embed_routing": os.getenv("EMBED_ROUTING"),
        }
    }


@router.get("/embeddings/test/{provider}")
async def test_provider(provider: str):
    """Test a specific embedding provider.
    
    Args:
        provider: Provider name to test (openai, voyage, cohere, fallback)
    """
    try:
        embedder = get_embedder(provider=provider)
        test_result = embedder.embed([f"Testing {provider} provider"])
        
        return {
            "provider": provider,
            "ok": True,
            "test_embedding_count": len(test_result),
            "embedding_dimension": embedder.get_embedding_dimension() if hasattr(embedder, 'get_embedding_dimension') else None,
            "sample_vector_length": len(test_result[0]) if test_result else 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "provider": provider,
                "ok": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )


@router.get("/embeddings/routing/{tag}")
async def test_routing(tag: str):
    """Test provider routing for a specific tag.
    
    Args:
        tag: Source tag to test routing for (e.g., 'obsidian', 'commits')
    """
    try:
        embedder = get_embedder_for(tag)
        test_result = embedder.embed([f"Testing routing for {tag} tag"])
        
        return {
            "tag": tag,
            "resolved_provider": embedder.__class__.__name__,
            "ok": True,
            "test_embedding_count": len(test_result),
            "embedding_dimension": embedder.get_embedding_dimension() if hasattr(embedder, 'get_embedding_dimension') else None,
            "sample_vector_length": len(test_result[0]) if test_result else 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "tag": tag,
                "ok": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )