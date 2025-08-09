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
    
    # Handle A/B testing (dual-write) if enabled
    ab_write_config = os.getenv("EMBED_AB_WRITE", "").strip()
    
    # Create base provider
    base_provider = _create_provider(provider, config)
    
    # Wrap with dual-write for A/B testing if configured
    if ab_write_config:
        try:
            from .providers.dualwrite import DualWriteEmbeddings
            base_provider = _create_dualwrite_provider(base_provider, ab_write_config)
        except ImportError:
            # Dual-write provider not available, use base
            pass
    
    # Wrap with cache if enabled (after dual-write)
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


def _create_dualwrite_provider(primary: EmbeddingProvider, config_str: str) -> EmbeddingProvider:
    """Create dual-write provider from configuration string.
    
    Args:
        primary: Primary embedding provider
        config_str: Configuration in multiple formats:
                   - "shadow_provider:percentage" e.g., "voyage:50" 
                   - "shadow_provider" (defaults to 100%)
                   - "primary,shadow" e.g., "openai,voyage" (100% dual-write)
        
    Returns:
        DualWriteEmbeddings instance
    """
    from .providers.dualwrite import DualWriteEmbeddings
    
    # Parse different config formats
    if "," in config_str:
        # Format: "primary,shadow" - 100% dual-write
        parts = config_str.split(",", 1)
        shadow_provider_name = parts[1].strip()
        percentage = 100.0
    elif ":" in config_str:
        # Format: "shadow:percentage"
        shadow_provider_name, percentage_str = config_str.split(":", 1)
        try:
            percentage = float(percentage_str)
        except ValueError:
            percentage = 100.0
    else:
        # Format: "shadow" (defaults to 100%)
        shadow_provider_name = config_str
        percentage = 100.0
    
    # Create shadow provider
    try:
        shadow_provider = _create_provider(shadow_provider_name.strip())
    except (ValueError, ImportError) as e:
        # Shadow provider not available, return primary only
        import logging
        logging.warning(f"Shadow provider {shadow_provider_name} not available: {e}")
        return primary
    
    # Create dual-write embeddings
    return DualWriteEmbeddings(
        primary=primary,
        shadow=shadow_provider,
        shadow_percentage=percentage,
        compare_results=True  # Enable comparison by default for A/B testing
    )


def _parse_routing(raw: str) -> Dict[str, str]:
    """Parse EMBED_ROUTING string into tag -> provider mapping.
    
    Format: "obsidian:openai,commits:voyage,default:openai"
    """
    mapping = {}
    if not raw:
        return mapping
    
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        tag, provider = pair.split(":", 1)
        mapping[tag.strip().lower()] = provider.strip().lower()
    
    return mapping


def get_embedder_for(source_tag: Optional[str] = None) -> EmbeddingProvider:
    """Get embedder for specific source tag with routing support.
    
    Args:
        source_tag: Source context tag (e.g., 'obsidian', 'commits', 'docs')
        
    Returns:
        Configured EmbeddingProvider instance
        
    Environment Variables:
        EMBED_ROUTING: Tag-based routing rules 
                      Format: "obsidian:openai,commits:voyage,default:openai"
    
    Examples:
        # Use OpenAI for Obsidian notes, Voyage for commits, fallback for others
        EMBED_ROUTING="obsidian:openai,commits:voyage,default:fallback"
        
        get_embedder_for("obsidian")  # -> OpenAI
        get_embedder_for("commits")   # -> Voyage  
        get_embedder_for("emails")    # -> Fallback (default)
    """
    routing = _parse_routing(os.getenv("EMBED_ROUTING", ""))
    if not routing:
        return get_embedder()  # No routing configured, use standard factory
    
    tag = (source_tag or "").lower()
    provider = routing.get(tag) or routing.get("default")
    
    if not provider:
        return get_embedder()  # No route found, use standard factory
        
    # Use internal factory with specific provider
    return _create_provider(provider)


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
        "dualwrite": True,  # Always available (built-in)
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


def get_embedder_ab() -> EmbeddingProvider:
    """Get embedder with A/B testing (dual-write) if configured.
    
    This is a specialized factory function that creates dual-write embeddings
    when EMBED_AB_WRITE is set, allowing you to shadow-write to a second provider
    while returning the primary provider's results.
    
    Returns:
        EmbeddingProvider instance (DualWriteEmbeddings if A/B testing enabled)
        
    Environment Variables:
        EMBED_AB_WRITE: Shadow provider configuration
                       Format: "provider:percentage" or "provider" (defaults to 100%)
                       Examples: "voyage:50", "cohere:25", "openai"
        
        EMBEDDING_PROVIDER: Primary provider (default: openai)
    
    Usage:
        # Configure A/B testing
        export EMBED_AB_WRITE="voyage:50"
        export EMBEDDING_PROVIDER="openai"
        
        # Use specialized A/B factory
        from integrations.embeddings_factory import get_embedder_ab
        embedder = get_embedder_ab()
        vectors = embedder.embed(texts)  # Returns primary, also sends to shadow
        
        # Check A/B statistics
        if hasattr(embedder, 'get_stats'):
            stats = embedder.get_stats()
            print(f"Shadow requests: {stats['shadow_requests']}")
    """
    ab_write_config = os.getenv("EMBED_AB_WRITE", "").strip()
    
    if not ab_write_config:
        # No A/B testing configured, return standard embedder
        return get_embedder()
    
    # Parse A/B configuration
    if "," in ab_write_config:
        # Format: "primary,shadow" - 100% dual-write
        primary_provider_name, shadow_provider_name = ab_write_config.split(",", 1)
        primary_provider_name = primary_provider_name.strip()
        shadow_provider_name = shadow_provider_name.strip()
        percentage = 100.0
    elif ":" in ab_write_config:
        # Format: "shadow:percentage"
        shadow_provider_name, percentage_str = ab_write_config.split(":", 1)
        shadow_provider_name = shadow_provider_name.strip()
        try:
            percentage = float(percentage_str)
        except ValueError:
            percentage = 100.0
        # Get primary provider from standard env var
        primary_provider_name = os.getenv("EMBEDDING_PROVIDER", "openai")
    else:
        # Format: "shadow" (defaults to 100%)
        shadow_provider_name = ab_write_config.strip()
        percentage = 100.0
        # Get primary provider from standard env var
        primary_provider_name = os.getenv("EMBEDDING_PROVIDER", "openai")
    
    # Create primary provider
    primary_provider = _create_provider(primary_provider_name.lower())
    
    # Create shadow provider
    try:
        shadow_provider = _create_provider(shadow_provider_name.strip().lower())
    except (ValueError, ImportError) as e:
        # Shadow provider not available, return primary only with warning
        import logging
        logging.warning(f"A/B testing disabled: shadow provider {shadow_provider_name} not available: {e}")
        return primary_provider
    
    # Create dual-write embeddings for A/B testing
    from .providers.dualwrite import DualWriteEmbeddings
    return DualWriteEmbeddings(
        primary=primary_provider,
        shadow=shadow_provider,
        shadow_percentage=percentage,
        compare_results=True  # Enable comparison for A/B testing
    )