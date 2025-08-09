#!/usr/bin/env python3
"""
Embedding Service
=================

Handles text embedding generation using OpenAI's embedding models.
Provides caching and batch processing for efficient embedding generation.
"""

import os
import hashlib
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and caching text embeddings"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", cache_dir: Optional[Path] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required: pip install openai")
        
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache_dir = cache_dir or Path("./data/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model dimensions
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        logger.info(f"Embedding service initialized with model {model}")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension for the current model"""
        return self.dimensions.get(self.model, 1536)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        content_hash = hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
        return content_hash
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a cache key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data.get('embedding')
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """Save embedding to cache"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'embedding': embedding,
                    'model': self.model,
                    'cached_at': str(Path().absolute()),
                }, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text.strip():
            return [0.0] * self.dimension
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding:
            return cached_embedding
        
        try:
            # Generate embedding via OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            self._save_to_cache(cache_key, embedding)
            
            logger.debug(f"Generated embedding for text ({len(text)} chars)")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.dimension
    
    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Check cache for each text in batch
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch):
                cache_key = self._get_cache_key(text)
                cached_embedding = self._load_from_cache(cache_key)
                
                if cached_embedding:
                    batch_embeddings.append(cached_embedding)
                else:
                    batch_embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=uncached_texts
                    )
                    
                    # Fill in the uncached embeddings
                    for idx, embedding_data in enumerate(response.data):
                        original_idx = uncached_indices[idx]
                        embedding = embedding_data.embedding
                        
                        batch_embeddings[original_idx] = embedding
                        
                        # Cache the result
                        cache_key = self._get_cache_key(uncached_texts[idx])
                        self._save_to_cache(cache_key, embedding)
                    
                    logger.info(f"Generated {len(uncached_texts)} embeddings in batch")
                    
                except Exception as e:
                    logger.error(f"Failed to generate batch embeddings: {e}")
                    # Fill remaining with zero vectors
                    for idx in uncached_indices:
                        if batch_embeddings[idx] is None:
                            batch_embeddings[idx] = [0.0] * self.dimension
            
            embeddings.extend(batch_embeddings)
            
            # Add small delay between batches to respect rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cached_embeddings": len(cache_files),
                "cache_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": str(self.cache_dir),
                "model": self.model,
                "dimension": self.dimension
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

# Simple embedding service for testing without OpenAI API
class MockEmbeddingService:
    """Mock embedding service for testing"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.model = "mock-embedding"
        logger.info("Mock embedding service initialized")
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate mock embedding (hash-based)"""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Create deterministic but varied embedding
        embedding = []
        for i in range(self.dimension):
            val = ((hash_int + i) % 1000) / 1000.0 - 0.5  # Range: -0.5 to 0.5
            embedding.append(val)
        
        return embedding
    
    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate mock embeddings for batch"""
        return [await self.embed_text(text) for text in texts]
    
    def clear_cache(self):
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cached_embeddings": 0,
            "cache_size_mb": 0,
            "cache_dir": "mock",
            "model": self.model,
            "dimension": self.dimension
        }