"""Factory for creating embedding providers based on configuration."""

import os
from typing import Optional, Dict, Any
from .embeddings_interfaces import EmbeddingProvider
from .embeddings import OpenAIEmbeddings  # Existing implementation

# Optional future imports (uncomment when implemented):
# from .providers.voyage import VoyageEmbeddings
# from .providers.cohere import CohereEmbeddings
# from .providers.anthropic import AnthropicEmbeddings
# from .providers.fallback import FallbackEmbeddings
# from .providers.cached import CachedEmbeddings


def get_embedder(
    provider: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> EmbeddingProvider:
    """Return the configured embedding provider. Defaults to OpenAI.
    
    Args:
        provider: Override provider name (defaults to EMBEDDING_PROVIDER env var)
        config: Optional configuration dictionary
        
    Returns:
        Configured EmbeddingProvider instance
        
    Env vars:
        EMBEDDING_PROVIDER: openai|voyage|cohere|anthropic|fallback|cached (default: openai)
        OPENAI_EMBED_MODEL: Model name for OpenAI (default: text-embedding-ada-002)
        OPENAI_API_KEY: OpenAI API key
        VOYAGE_EMBED_MODEL: Model name for Voyage (default: voyage-3)
        VOYAGE_API_KEY: Voyage API key
        COHERE_EMBED_MODEL: Model name for Cohere (default: embed-english-v3.0)
        COHERE_API_KEY: Cohere API key
        ANTHROPIC_EMBED_MODEL: Model name for Anthropic
        ANTHROPIC_API_KEY: Anthropic API key
        EMBEDDING_CACHE_ENABLED: Enable caching layer (true/false)
        EMBEDDING_FALLBACK_PROVIDER: Backup provider for fallback mode
    """
    # Determine provider
    if provider is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    
    # Handle caching wrapper if enabled
    cache_enabled = os.getenv("EMBEDDING_CACHE_ENABLED", "false").lower() == "true"
    
    # Create base provider
    base_provider = _create_provider(provider, config)
    
    # Wrap with cache if enabled
    if cache_enabled:
        try:
            from .providers.cached import CachedEmbeddings
            return CachedEmbeddings(base_provider)
        except ImportError:
            # Cache provider not available, use base
            pass
    
    return base_provider


def _create_provider(provider: str, config: Optional[Dict[str, Any]] = None) -> EmbeddingProvider:
    """Create a specific embedding provider.
    
    Args:
        provider: Provider name
        config: Optional configuration
        
    Returns:
        Configured provider instance
        
    Raises:
        ValueError: If provider is unknown
    """
    if provider == "openai":
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
        api_key = os.getenv("OPENAI_API_KEY")
        # Allow test key for testing
        if not api_key or api_key == "test-key":
            api_key = "test-key"  # Will be mocked in tests
        return OpenAIEmbeddings(model=model, api_key=api_key)
    
    elif provider == "voyage":
        try:
            from .providers.voyage import VoyageEmbeddings
            model = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3")
            api_key = os.getenv("VOYAGE_API_KEY")
            return VoyageEmbeddings(model=model, api_key=api_key)
        except ImportError:
            raise ValueError(f"Voyage provider not installed. Install voyage-ai package.")
    
    elif provider == "cohere":
        try:
            from .providers.cohere import CohereEmbeddings
            model = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
            api_key = os.getenv("COHERE_API_KEY")
            return CohereEmbeddings(model=model, api_key=api_key)
        except ImportError:
            raise ValueError(f"Cohere provider not installed. Install cohere package.")
    
    elif provider == "anthropic":
        try:
            from .providers.anthropic import AnthropicEmbeddings
            model = os.getenv("ANTHROPIC_EMBED_MODEL", "claude-embed")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            return AnthropicEmbeddings(model=model, api_key=api_key)
        except ImportError:
            raise ValueError(f"Anthropic provider not installed.")
    
    elif provider == "fallback":
        try:
            from .providers.fallback import FallbackEmbeddings
            # Create primary and backup providers
            primary_name = os.getenv("EMBEDDING_PRIMARY_PROVIDER", "openai")
            backup_name = os.getenv("EMBEDDING_FALLBACK_PROVIDER", "cohere")
            
            primary = _create_provider(primary_name)
            backup = _create_provider(backup_name)
            
            return FallbackEmbeddings(primary=primary, backup=backup)
        except ImportError:
            raise ValueError(f"Fallback provider not available.")
    
    elif provider == "cached":
        # Cached provider wraps another provider
        try:
            from .providers.cached import CachedEmbeddings
            base_provider_name = os.getenv("EMBEDDING_BASE_PROVIDER", "openai")
            base = _create_provider(base_provider_name)
            return CachedEmbeddings(base)
        except ImportError:
            raise ValueError(f"Cached provider not available.")
    
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER={provider!r}. "
            f"Supported: openai, voyage, cohere, anthropic, fallback, cached"
        )


def get_available_providers() -> Dict[str, bool]:
    """Check which embedding providers are available.
    
    Returns:
        Dictionary mapping provider names to availability status
    """
    providers = {
        "openai": True,  # Always available (built-in)
        "voyage": False,
        "cohere": False,
        "anthropic": False,
        "fallback": False,
        "cached": False,
    }
    
    # Check optional providers
    try:
        import voyageai
        providers["voyage"] = True
    except ImportError:
        pass
    
    try:
        import cohere
        providers["cohere"] = True
    except ImportError:
        pass
    
    try:
        from .providers.anthropic import AnthropicEmbeddings
        providers["anthropic"] = True
    except ImportError:
        pass
    
    try:
        from .providers.fallback import FallbackEmbeddings
        providers["fallback"] = True
    except ImportError:
        pass
    
    try:
        from .providers.cached import CachedEmbeddings
        providers["cached"] = True
    except ImportError:
        pass
    
    return providers