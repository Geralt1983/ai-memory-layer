# Embeddings: Provider-Swappable Layer

The AI Memory Layer now supports multiple embedding providers with automatic fallback and caching capabilities. This allows you to switch between providers without changing your code, optimize costs, and ensure high availability.

## Features

- **Protocol-based Interface**: Clean Python Protocol interface for all providers
- **Multiple Providers**: OpenAI (default), Voyage, Cohere, with more coming
- **Automatic Fallback**: High availability with primary/backup provider switching
- **Environment Configuration**: Switch providers via environment variables
- **Backward Compatible**: Existing code continues to work unchanged
- **Caching Support**: Optional embedding cache to reduce API calls
- **Factory Pattern**: Clean provider instantiation with configuration

## Quick Start

### Using the Factory (Recommended)

```python
from integrations.embeddings_factory import get_embedder

# Automatically uses configured provider (defaults to OpenAI)
embedder = get_embedder()
vectors = embedder.embed(["text1", "text2", "text3"])
```

### Backward Compatible

```python
# Existing code continues to work
from integrations.embeddings import OpenAIEmbeddings
embedder = OpenAIEmbeddings(api_key="sk-...")
```

## Configuration

### Environment Variables

```bash
# Provider Selection
EMBEDDING_PROVIDER=openai  # Options: openai, voyage, cohere, fallback, cached
                           # Default: openai

# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-ada-002  # Default model

# Voyage Configuration (when available)
VOYAGE_API_KEY=...
VOYAGE_EMBED_MODEL=voyage-3  # Options: voyage-3, voyage-3-lite, voyage-code-3

# Cohere Configuration (when available)
COHERE_API_KEY=...
COHERE_EMBED_MODEL=embed-english-v3.0  # Options: embed-english-v3.0, embed-multilingual-v3.0

# Fallback Configuration
EMBEDDING_PRIMARY_PROVIDER=openai
EMBEDDING_FALLBACK_PROVIDER=cohere

# Advanced: Tag-based routing
EMBED_ROUTING="obsidian:openai,commits:voyage,docs:cohere,default:fallback"

# Caching
EMBEDDING_CACHE_ENABLED=true  # Enable caching layer
EMBED_CACHE=.cache/embeddings.json  # Cache file path
```

### Python Configuration

```python
from integrations.embeddings_factory import get_embedder, get_embedder_for

# Override provider programmatically
embedder = get_embedder(provider="openai")

# Tag-based routing (requires EMBED_ROUTING environment variable)
obsidian_embedder = get_embedder_for("obsidian")  # Routes to configured provider
commit_embedder = get_embedder_for("commits")     # Routes to different provider
default_embedder = get_embedder_for("unknown")    # Uses default route

# With custom configuration
config = {
    "model": "text-embedding-3-large",
    "dimensions": 3072
}
embedder = get_embedder(provider="openai", config=config)
```

## Provider Details

### OpenAI (Default)

The default provider with excellent performance and wide model selection.

```python
# Models available:
# - text-embedding-ada-002 (1536 dims) - Most cost-effective
# - text-embedding-3-small (1536 dims) - Better performance
# - text-embedding-3-large (3072 dims) - Best performance
```

### Voyage (✅ Available with lazy import)

State-of-the-art embeddings optimized for retrieval. Auto-installs when needed.

```python
# Models available:
# - voyage-3 (1024 dims) - Latest general-purpose
# - voyage-3-lite (512 dims) - Faster, smaller
# - voyage-code-3 (1536 dims) - Optimized for code
# - voyage-finance-2 (1024 dims) - Domain-specific
# - voyage-law-2 (1024 dims) - Legal documents
```

### Cohere (Stub - Implement when needed)

Multilingual embeddings with compression support.

```python
# Models available:
# - embed-english-v3.0 (1024 dims) - English
# - embed-multilingual-v3.0 (1024 dims) - 100+ languages
# - embed-english-light-v3.0 (384 dims) - Smaller English
# - embed-multilingual-light-v3.0 (384 dims) - Smaller multilingual
```

## Advanced Features

### A/B Testing with Dual-Write

Shadow-write embedding requests to a second provider while returning the primary provider's results, enabling quality and latency comparison without changing runtime behavior:

```bash
# Configure dual-write A/B testing
export EMBED_AB_WRITE="openai,voyage"     # 100% dual-write: primary=openai, shadow=voyage
# OR
export EMBED_AB_WRITE="voyage:50"         # 50% shadow traffic to voyage
# OR
export EMBED_AB_WRITE="cohere:25"         # 25% shadow traffic to cohere
```

```python
from integrations.embeddings_factory import get_embedder_ab

# Use specialized A/B testing factory
embedder = get_embedder_ab()     # Returns dual-write wrapper if EMBED_AB_WRITE set
vectors = embedder.embed(texts)  # Returns primary result, also sends to shadow

# Check A/B testing statistics
if hasattr(embedder, 'get_stats'):
    stats = embedder.get_stats()
    print(f"Primary success rate: {stats['primary_success_rate']:.2%}")
    print(f"Shadow requests: {stats['shadow_requests']}")
    print(f"Average similarity: {stats.get('avg_cosine_similarity', 'N/A')}")
    print(f"Primary avg time: {stats.get('primary_avg_time', 0):.3f}s")
    print(f"Shadow avg time: {stats.get('shadow_avg_time', 0):.3f}s")

# Progressive rollout: start with 10%, increase to 100%
if hasattr(embedder, 'set_shadow_percentage'):
    embedder.set_shadow_percentage(10.0)   # Start with 10% shadow traffic
    # ... monitor for issues ...
    embedder.set_shadow_percentage(50.0)   # Increase to 50%
    # ... continue monitoring ...
    embedder.set_shadow_percentage(100.0)  # Full dual-write for comparison
```

**Configuration Formats:**
- `"primary,shadow"` - 100% dual-write with explicit primary and shadow providers
- `"shadow:percentage"` - Shadow provider with traffic percentage (primary from EMBEDDING_PROVIDER)
- `"shadow"` - Shadow provider with 100% traffic (primary from EMBEDDING_PROVIDER)

**Use Cases:**
- **Quality comparison**: Compare embedding quality between providers
- **Performance testing**: Measure latency differences under production load
- **Migration testing**: Validate new provider before switching
- **Cost analysis**: Compare API costs and rate limits

### Health Helper (Richer Output)
Expose an internal endpoint that returns the active embedding status, including A/B:
```python
from integrations.ops.embeddings_health import check_embeddings_health

@router.get("/internal/embeddings/health")
def embeddings_health():
    return check_embeddings_health()
```
Sample response (A/B):
```json
{
  "ok": true,
  "mode": "ab",
  "provider_env": "openai",
  "active": {"provider": "openai", "model": "text-embedding-3-small", "stats": {...}},
  "shadow": {"provider": "voyage", "model": "voyage-3", "stats": {...}}
}
```

### Tag-Based Provider Routing

Route different content types to different providers for optimal performance:

```bash
# Configure routing rules
export EMBED_ROUTING="obsidian:openai,commits:voyage,docs:cohere,default:fallback"
```

```python
from integrations.embeddings_factory import get_embedder_for

# Each content type gets routed to its optimal provider
obsidian_notes = get_embedder_for("obsidian")    # -> OpenAI
code_commits = get_embedder_for("commits")       # -> Voyage  
documentation = get_embedder_for("docs")         # -> Cohere
unknown_content = get_embedder_for("emails")     # -> Fallback (default)
```

**Use Cases:**
- **Obsidian notes** → OpenAI (excellent for general text)
- **Code commits** → Voyage (optimized for technical content)  
- **Documentation** → Cohere (strong multilingual support)
- **Default fallback** → High availability with automatic switching

### Operational Monitoring

The system includes monitoring endpoints for production deployments:

```bash
# Check active provider health
curl /internal/embeddings/health
# {"provider":"openai","ok":true,"embedding_dimension":1536}

# List all available providers and configuration  
curl /internal/embeddings/providers
# {"active_provider":"openai","available_providers":{"openai":true,"voyage":false,...}}

# Test specific provider
curl /internal/embeddings/test/voyage
# {"provider":"voyage","ok":true,"test_embedding_count":1}

# Test routing for specific tag
curl /internal/embeddings/routing/obsidian
# {"tag":"obsidian","resolved_provider":"OpenAIEmbeddings","ok":true}
```

## High Availability with Fallback

Ensure your application stays online even when providers fail:

```python
# Configure via environment
EMBEDDING_PROVIDER=fallback
EMBEDDING_PRIMARY_PROVIDER=openai
EMBEDDING_FALLBACK_PROVIDER=cohere

# Or programmatically
from integrations.providers.fallback import FallbackEmbeddings
from integrations.embeddings import OpenAIEmbeddings
from integrations.providers.cohere import CohereEmbeddings

primary = OpenAIEmbeddings()
backup = CohereEmbeddings()
embedder = FallbackEmbeddings(primary=primary, backup=backup)

# Automatically uses backup if primary fails
vectors = embedder.embed(texts)

# Check statistics
stats = embedder.get_stats()
print(f"Primary failures: {stats['primary_failures']}")
print(f"Backup uses: {stats['backup_uses']}")
```

### Multi-Provider Fallback Chain

For maximum reliability with multiple backup providers:

```python
from integrations.providers.fallback import MultiProviderFallback

providers = [
    OpenAIEmbeddings(),      # Primary
    VoyageEmbeddings(),      # First backup
    CohereEmbeddings(),      # Second backup
]

embedder = MultiProviderFallback(providers)
vectors = embedder.embed(texts)  # Tries each provider in order
```

## Caching

Reduce API calls and costs with the built-in cache:

```python
# Enable via environment
EMBEDDING_CACHE_ENABLED=true

# Cache is automatically used by the factory
embedder = get_embedder()  # Will use cache if enabled

# Or use cache directly
from core.embedding_cache import get_embedding_cache

cache = get_embedding_cache()
cached_vector = cache.get("text to embed")
if cached_vector is None:
    # Not in cache, compute and store
    vector = embedder.embed_text("text to embed")
    cache.put("text to embed", vector)
```

## Migration Guide

### From Direct OpenAI Usage

Before:
```python
import openai
client = openai.OpenAI(api_key="sk-...")
response = client.embeddings.create(model="text-embedding-ada-002", input=texts)
vectors = [d.embedding for d in response.data]
```

After:
```python
from integrations.embeddings_factory import get_embedder
embedder = get_embedder()  # Uses env vars for configuration
vectors = embedder.embed(texts)
```

### From Existing Memory Engine

No changes needed! The MemoryEngine already uses the abstraction:

```python
# This continues to work unchanged
from core.memory_engine import MemoryEngine
from integrations.embeddings import OpenAIEmbeddings

engine = MemoryEngine(
    embedding_provider=OpenAIEmbeddings(),
    # ...
)

# To use the factory:
from integrations.embeddings_factory import get_embedder

engine = MemoryEngine(
    embedding_provider=get_embedder(),  # Now provider-agnostic!
    # ...
)
```

## Testing

Run the embedding tests to verify your configuration:

```bash
# Test embedding contract (all providers should pass)
pytest tests/test_embeddings_contract.py -v

# Test fallback functionality
pytest tests/test_embeddings_fallback.py -v

# Test with different providers
EMBEDDING_PROVIDER=openai pytest tests/test_embeddings_contract.py
EMBEDDING_PROVIDER=fallback pytest tests/test_embeddings_fallback.py
```

## Adding New Providers

To add a new embedding provider:

1. Create a new file in `integrations/providers/your_provider.py`
2. Implement the `EmbeddingProvider` protocol:

```python
from typing import List, Optional
from ..embeddings_interfaces import EmbeddingProvider

class YourProviderEmbeddings:
    def __init__(self, model: str = "default-model", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("YOUR_PROVIDER_API_KEY")
        # Initialize client
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Implement batch embedding
        pass
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        # Implement single text embedding
        pass
    
    def get_embedding_dimension(self) -> int:
        # Return dimension for this model
        pass
```

3. Add to factory in `integrations/embeddings_factory.py`:

```python
elif provider == "your_provider":
    from .providers.your_provider import YourProviderEmbeddings
    return YourProviderEmbeddings(model=os.getenv("YOUR_PROVIDER_MODEL"))
```

4. Add tests to verify the contract is maintained

## Performance Considerations

- **Batching**: Send multiple texts in one call for better performance
- **Caching**: Enable cache to avoid re-embedding identical texts
- **Model Selection**: Balance quality vs speed/cost
  - Smaller models (384-512 dims) for high-volume, low-latency
  - Larger models (1536-3072 dims) for best quality
- **Fallback Overhead**: Minimal - only activates on primary failure

## Troubleshooting

### Provider Not Available

```python
from integrations.embeddings_factory import get_available_providers
print(get_available_providers())
# {'openai': True, 'voyage': False, 'cohere': False, ...}
```

### API Key Issues

```bash
# Check environment
echo $OPENAI_API_KEY
echo $EMBEDDING_PROVIDER

# Set in code if needed
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Dimension Mismatch

Different providers use different dimensions. When switching providers with existing data:

```python
# Check current dimension
embedder = get_embedder()
print(f"Dimension: {embedder.get_embedding_dimension()}")

# Ensure vector store matches
vector_store = FaissVectorStore(dimension=embedder.get_embedding_dimension())
```

## Future Enhancements

- [ ] Anthropic Claude embeddings (when available)
- [ ] Local embedding models (Sentence Transformers)
- [ ] Embedding compression for storage optimization
- [ ] A/B testing framework for provider comparison
- [ ] Cost tracking and optimization
- [ ] Rate limit handling with automatic backoff
- [ ] Embedding versioning for model updates

## Support

For issues or questions about embeddings:
1. Check the provider's documentation
2. Verify API keys and network connectivity
3. Enable debug logging: `LOG_LEVEL=DEBUG`
4. Open an issue with provider details and error messages