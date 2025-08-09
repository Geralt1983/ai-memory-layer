"""Helper function to create embedding provider using the factory pattern.

This module provides a backward-compatible way to create embedding providers
while allowing new code to use the factory pattern.
"""

import os
from typing import Optional
from .embeddings_factory import get_embedder
from .embeddings import OpenAIEmbeddings


def create_embedding_provider(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None
):
    """Create an embedding provider with backward compatibility.
    
    This function maintains backward compatibility while using the new factory.
    
    Args:
        provider: Provider name (if None, uses env var or defaults to openai)
        api_key: API key (if None, uses env var)
        model: Model name (if None, uses env var or provider default)
        
    Returns:
        EmbeddingProvider instance
    """
    # If specific parameters are provided, create directly for backward compat
    if provider is None and api_key is not None:
        # Legacy code path - direct OpenAI creation
        if model is None:
            model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
        return OpenAIEmbeddings(api_key=api_key, model=model)
    
    # Use factory for new code path
    return get_embedder(provider=provider)