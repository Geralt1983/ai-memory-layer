"""
Embedding Cache Module - Optimizes performance by caching embeddings for repeated queries
"""

from typing import Dict, List, Optional, Tuple
import hashlib
import time
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
from dataclasses import dataclass, asdict


@dataclass
class CachedEmbedding:
    """Represents a cached embedding with metadata"""
    query: str
    embedding: List[float]
    timestamp: datetime
    hit_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'query': self.query,
            'embedding': self.embedding,
            'timestamp': self.timestamp.isoformat(),
            'hit_count': self.hit_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(
            query=data['query'],
            embedding=data['embedding'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            hit_count=data.get('hit_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None
        )


class EmbeddingCache:
    """
    Thread-safe LRU cache for embeddings with TTL and persistence support
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_hours: int = 24,
        persist_path: Optional[str] = None,
        auto_save_interval: int = 300  # Auto-save every 5 minutes
    ):
        """
        Initialize the embedding cache
        
        Args:
            max_size: Maximum number of cached embeddings
            ttl_hours: Time-to-live for cached embeddings in hours
            persist_path: Optional path to persist cache to disk
            auto_save_interval: Auto-save interval in seconds
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.persist_path = persist_path
        self.auto_save_interval = auto_save_interval
        
        # Thread-safe cache storage
        self._cache: Dict[str, CachedEmbedding] = {}
        self._lock = threading.RLock()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Load persistent cache if available
        if persist_path:
            self._load_cache()
            
        # Start auto-save thread if persistence is enabled
        if persist_path and auto_save_interval > 0:
            self._start_auto_save()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a unique cache key for the text"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding if found and valid, None otherwise
        """
        with self._lock:
            cache_key = self._get_cache_key(text)
            
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                
                # Check if cache entry is still valid (not expired)
                if datetime.now(timezone.utc) - cached.timestamp < self.ttl:
                    # Update cache statistics
                    cached.hit_count += 1
                    cached.last_accessed = datetime.now(timezone.utc)
                    self.hits += 1
                    
                    # Move to end for LRU (by reinserting)
                    del self._cache[cache_key]
                    self._cache[cache_key] = cached
                    
                    return cached.embedding
                else:
                    # Expired entry, remove it
                    del self._cache[cache_key]
                    self.evictions += 1
            
            self.misses += 1
            return None
    
    def put(self, text: str, embedding: List[float]) -> None:
        """
        Cache an embedding
        
        Args:
            text: Original text
            embedding: Embedding vector to cache
        """
        with self._lock:
            cache_key = self._get_cache_key(text)
            
            # Check if we need to evict entries (LRU)
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                # Remove oldest entry (first in dict)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self.evictions += 1
            
            # Add or update cache entry
            self._cache[cache_key] = CachedEmbedding(
                query=text[:100],  # Store first 100 chars for debugging
                embedding=embedding,
                timestamp=datetime.now(timezone.utc),
                hit_count=0,
                last_accessed=None
            )
    
    def clear(self) -> None:
        """Clear all cached embeddings"""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': f"{hit_rate:.2f}%",
                'total_requests': total_requests
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from cache"""
        with self._lock:
            current_time = datetime.now(timezone.utc)
            expired_keys = [
                key for key, cached in self._cache.items()
                if current_time - cached.timestamp >= self.ttl
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self.evictions += 1
            
            return len(expired_keys)
    
    def _save_cache(self) -> None:
        """Save cache to disk"""
        if not self.persist_path:
            return
            
        with self._lock:
            try:
                cache_data = {
                    'version': '1.0',
                    'stats': self.get_stats(),
                    'entries': {
                        key: cached.to_dict()
                        for key, cached in self._cache.items()
                    }
                }
                
                # Create directory if needed
                Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Write to temporary file first, then rename (atomic operation)
                temp_path = f"{self.persist_path}.tmp"
                with open(temp_path, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                
                # Atomic rename
                Path(temp_path).rename(self.persist_path)
                
            except Exception as e:
                print(f"Error saving embedding cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk"""
        if not self.persist_path or not Path(self.persist_path).exists():
            return
            
        with self._lock:
            try:
                with open(self.persist_path, 'r') as f:
                    cache_data = json.load(f)
                
                # Restore cache entries
                for key, entry_data in cache_data.get('entries', {}).items():
                    cached = CachedEmbedding.from_dict(entry_data)
                    
                    # Only restore non-expired entries
                    if datetime.now(timezone.utc) - cached.timestamp < self.ttl:
                        self._cache[key] = cached
                
                # Restore statistics
                stats = cache_data.get('stats', {})
                self.hits = stats.get('hits', 0)
                self.misses = stats.get('misses', 0)
                self.evictions = stats.get('evictions', 0)
                
                print(f"Loaded {len(self._cache)} cached embeddings from disk")
                
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
    
    def _start_auto_save(self) -> None:
        """Start background thread for auto-saving cache"""
        def auto_save_worker():
            while True:
                time.sleep(self.auto_save_interval)
                self._save_cache()
                # Also cleanup expired entries periodically
                expired = self.cleanup_expired()
                if expired > 0:
                    print(f"Cleaned up {expired} expired cache entries")
        
        # Start as daemon thread so it doesn't prevent program exit
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
    
    def save(self) -> None:
        """Manually save cache to disk"""
        self._save_cache()


# Global cache instance (singleton pattern)
_global_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(
    max_size: int = 1000,
    ttl_hours: int = 24,
    persist_path: Optional[str] = "data/embedding_cache.json"
) -> EmbeddingCache:
    """
    Get or create the global embedding cache instance
    
    Args:
        max_size: Maximum number of cached embeddings
        ttl_hours: Time-to-live for cached embeddings
        persist_path: Path to persist cache
        
    Returns:
        Global EmbeddingCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = EmbeddingCache(
            max_size=max_size,
            ttl_hours=ttl_hours,
            persist_path=persist_path
        )
    
    return _global_cache