"""
Cached Embedding Provider - Wraps any embedding provider with caching capability
"""

from typing import List, Union, Optional
import numpy as np
from core.embedding_cache import get_embedding_cache, EmbeddingCache
from integrations.embeddings import EmbeddingProvider
from core.logging_config import get_logger


class CachedEmbeddingProvider(EmbeddingProvider):
    """
    Wrapper for any embedding provider that adds caching capability
    """
    
    def __init__(
        self,
        base_provider: EmbeddingProvider,
        cache: Optional[EmbeddingCache] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize cached embedding provider
        
        Args:
            base_provider: The underlying embedding provider to wrap
            cache: Optional custom cache instance (uses global cache if None)
            cache_enabled: Whether caching is enabled
        """
        self.base_provider = base_provider
        self.cache = cache or get_embedding_cache()
        self.cache_enabled = cache_enabled
        self.logger = get_logger("cached_embeddings")
        
        self.logger.info(
            "Cached embedding provider initialized",
            extra={
                "cache_enabled": cache_enabled,
                "cache_size": len(self.cache._cache),
                "provider": type(base_provider).__name__
            }
        )
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embedding with caching
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Convert list to string if needed
        if isinstance(text, list):
            text = " ".join(text)
        
        # Check cache first if enabled
        if self.cache_enabled:
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                self.logger.debug(
                    "Cache hit for embedding",
                    extra={"text_length": len(text), "cache_stats": self.cache.get_stats()}
                )
                return np.array(cached_embedding)
        
        # Cache miss - generate embedding
        self.logger.debug(
            "Cache miss, generating new embedding",
            extra={"text_length": len(text)}
        )
        
        # Generate embedding using base provider
        embedding = self.base_provider.embed_text(text)
        
        # Cache the result if enabled
        if self.cache_enabled:
            # Convert to list if it's a numpy array
            if hasattr(embedding, 'tolist'):
                self.cache.put(text, embedding.tolist())
            else:
                self.cache.put(text, embedding)
            self.logger.debug(
                "Cached new embedding",
                extra={"text_length": len(text), "cache_size": len(self.cache._cache)}
            )
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate batch embeddings with caching
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        texts_to_generate = []
        text_indices = []
        
        # Check cache for each text if enabled
        if self.cache_enabled:
            for i, text in enumerate(texts):
                cached_embedding = self.cache.get(text)
                if cached_embedding is not None:
                    embeddings.append((i, np.array(cached_embedding)))
                else:
                    texts_to_generate.append(text)
                    text_indices.append(i)
            
            cache_hits = len(embeddings)
            cache_misses = len(texts_to_generate)
            
            self.logger.debug(
                "Batch cache check completed",
                extra={
                    "total_texts": len(texts),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "cache_stats": self.cache.get_stats()
                }
            )
        else:
            texts_to_generate = texts
            text_indices = list(range(len(texts)))
        
        # Generate embeddings for cache misses
        if texts_to_generate:
            new_embeddings = self.base_provider.embed_batch(texts_to_generate)
            
            # Cache the new embeddings if enabled
            if self.cache_enabled:
                for text, embedding in zip(texts_to_generate, new_embeddings):
                    # Convert to list if it's a numpy array
                    if hasattr(embedding, 'tolist'):
                        self.cache.put(text, embedding.tolist())
                    else:
                        self.cache.put(text, embedding)
            
            # Add to results with original indices
            for idx, embedding in zip(text_indices, new_embeddings):
                embeddings.append((idx, embedding))
        
        # Sort by original index to maintain order
        embeddings.sort(key=lambda x: x[0])
        
        # Return just the embeddings in original order
        return [emb for _, emb in embeddings]
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable caching"""
        self.cache_enabled = enabled
        self.logger.info(f"Cache {'enabled' if enabled else 'disabled'}")
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self):
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def save_cache(self):
        """Save cache to disk"""
        self.cache.save()
        self.logger.info("Cache saved to disk")