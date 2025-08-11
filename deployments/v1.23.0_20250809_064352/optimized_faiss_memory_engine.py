#!/usr/bin/env python3
"""
Optimized FAISS Memory Engine - 2025 Best Practices
Implements precomputed index loading, query caching, and advanced FAISS optimizations
"""

import json
import os
import pickle
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import hashlib

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None

from core.memory_engine import Memory, MemoryEngine
from integrations.embeddings import EmbeddingProvider
from core.logging_config import get_logger, monitor_performance

logger = get_logger(__name__)

class QueryEmbeddingCache:
    """
    LRU cache for query embeddings to avoid recomputation
    Follows 2025 best practices for embedding caching
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str) -> str:
        """Create hash key for query"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def get(self, query: str) -> Optional[np.ndarray]:
        """Get cached embedding if exists"""
        key = self._hash_query(query)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, query: str, embedding: np.ndarray):
        """Cache embedding with LRU eviction"""
        key = self._hash_query(query)
        
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
        
        self.cache[key] = embedding.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

class OptimizedFAISSMemoryEngine(MemoryEngine):
    """
    Optimized Memory Engine implementing 2025 FAISS best practices:
    - Precomputed index loading (no embedding regeneration)
    - Query embedding caching
    - HNSW/IVF-PQ indexing for speed
    - Metadata filtering and importance boosting
    - Batched operations for better performance
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        persist_path: Optional[str] = None,
        index_path: Optional[str] = None,
        cache_size: int = 1000,
        enable_query_cache: bool = True,
        use_hnsw: bool = True,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 100
    ):
        """
        Initialize optimized FAISS memory engine
        
        Args:
            embedding_provider: For new query embeddings only
            persist_path: Path to memory JSON file
            index_path: Path to precomputed FAISS index (without extension)
            cache_size: Size of query embedding cache
            enable_query_cache: Whether to cache query embeddings
            use_hnsw: Use HNSW index for better performance
            hnsw_m: HNSW parameter - number of connections
            hnsw_ef_construction: HNSW construction parameter
            hnsw_ef_search: HNSW search parameter
        """
        # Don't call parent __init__ to avoid vector_store creation
        self.embedding_provider = embedding_provider
        self.persist_path = persist_path
        self.memories: List[Memory] = []
        
        # FAISS optimization parameters
        self.index_path = index_path
        self.use_hnsw = use_hnsw and FAISS_AVAILABLE
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        
        # Query caching
        self.query_cache = QueryEmbeddingCache(cache_size) if enable_query_cache else None
        
        # FAISS components
        self.faiss_index = None
        self.index_to_memory_map: Dict[int, int] = {}  # FAISS index -> memory index
        self.dimension = 1536  # OpenAI ada-002 dimension
        
        logger.info(
            "OptimizedFAISSMemoryEngine initialized",
            extra={
                "use_hnsw": self.use_hnsw,
                "cache_enabled": enable_query_cache,
                "cache_size": cache_size,
                "index_path": index_path
            }
        )
    
    @monitor_performance("load_precomputed_index")
    def load_precomputed_index(self, memory_json_path: str, index_base_path: str) -> bool:
        """
        Load precomputed FAISS index and memories following 2025 best practices
        
        Args:
            memory_json_path: Path to memory JSON file
            index_base_path: Base path for FAISS files (without extensions)
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Loading precomputed FAISS index from {index_base_path}")
        
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available - cannot load precomputed index")
            return False
        
        index_file = f"{index_base_path}.index"
        metadata_file = f"{index_base_path}.pkl"
        
        # Check if precomputed files exist
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            logger.warning(f"Precomputed index files not found: {index_file}, {metadata_file}")
            return False
        
        try:
            # 1. Load FAISS index directly (no embedding regeneration)
            logger.info("Loading FAISS index from disk...")
            self.faiss_index = faiss.read_index(index_file)
            
            # Configure HNSW search parameters if using HNSW
            if self.use_hnsw and hasattr(self.faiss_index, 'hnsw'):
                self.faiss_index.hnsw.ef = self.hnsw_ef_search
                logger.info(f"HNSW search ef set to {self.hnsw_ef_search}")
            
            logger.info(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")
            
            # 2. Load metadata and memory mapping
            logger.info("Loading index metadata...")
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Extract mapping information
            if 'index_to_memory_map' in metadata:
                self.index_to_memory_map = metadata['index_to_memory_map']
            else:
                # Create default mapping (assume 1:1)
                self.index_to_memory_map = {i: i for i in range(self.faiss_index.ntotal)}
            
            logger.info(f"Index mapping loaded: {len(self.index_to_memory_map)} entries")
            
            # 3. Load memories from JSON
            logger.info(f"Loading memories from {memory_json_path}")
            with open(memory_json_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Convert to Memory objects
            self.memories = []
            for i, mem_data in enumerate(memory_data):
                try:
                    timestamp = self._parse_timestamp(mem_data.get('timestamp'))
                    
                    # Store enhanced fields in metadata for compatibility
                    enhanced_metadata = {
                        **mem_data.get('metadata', {}),
                        'role': mem_data.get('role', 'user'),
                        'thread_id': mem_data.get('thread_id'),
                        'title': mem_data.get('title'),
                        'type': mem_data.get('type', 'history'),
                        'importance': mem_data.get('importance', 1.0),
                        'source': 'chatgpt_import'
                    }
                    
                    memory = Memory(
                        content=mem_data.get('content', ''),
                        embedding=None,  # Use FAISS index instead
                        metadata=enhanced_metadata,
                        timestamp=timestamp,
                        relevance_score=0.0
                    )
                    
                    self.memories.append(memory)
                    
                except Exception as e:
                    logger.error(f"Error processing memory {i}: {e}")
                    continue
            
            logger.info(f"Loaded {len(self.memories)} memories")
            
            # Verify alignment
            if len(self.memories) != self.faiss_index.ntotal:
                logger.warning(
                    f"Memory count mismatch: {len(self.memories)} memories vs "
                    f"{self.faiss_index.ntotal} FAISS vectors"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading precomputed index: {e}")
            return False
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp string efficiently"""
        if not timestamp_str:
            return datetime.now()
        
        try:
            if 'T' in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).replace(tzinfo=None)
            else:
                return datetime.fromtimestamp(float(timestamp_str))
        except:
            return datetime.now()
    
    @monitor_performance("get_cached_embedding")
    def get_cached_embedding(self, query: str) -> np.ndarray:
        """
        Get query embedding with caching to avoid recomputation
        Implements 2025 best practice for embedding caching
        """
        if not self.embedding_provider:
            raise ValueError("No embedding provider available")
        
        # Check cache first
        if self.query_cache:
            cached_embedding = self.query_cache.get(query)
            if cached_embedding is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_embedding
        
        # Generate new embedding
        logger.debug(f"Cache miss - generating embedding for: {query[:50]}...")
        embedding = self.embedding_provider.embed_text(query)
        
        if embedding is None:
            raise ValueError(f"Failed to generate embedding for query: {query}")
        
        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Cache the result
        if self.query_cache:
            self.query_cache.put(query, embedding)
        
        return embedding
    
    @monitor_performance("optimized_search")
    def search_memories(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
        filter_by_type: Optional[List[str]] = None,
        filter_by_role: Optional[List[str]] = None,
        importance_boost: float = 0.5,
        age_decay_days: float = 30.0
    ) -> List[Memory]:
        """
        Optimized memory search with precomputed FAISS index
        Implements 2025 best practices for retrieval with metadata filtering
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum relevance score
            filter_by_type: Filter by memory types
            filter_by_role: Filter by memory roles
            importance_boost: Boost factor for importance (0.5 = 50% boost)
            age_decay_days: Days for age decay half-life
            
        Returns:
            List of relevant memories with optimized scoring
        """
        if not self.memories:
            logger.warning("No memories available for search")
            return []
        
        if not self.faiss_index:
            logger.error("FAISS index not loaded")
            return []
        
        try:
            # Get cached query embedding
            query_embedding = self.get_cached_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search with FAISS (get more results for filtering and re-ranking)
            search_k = min(k * 3, self.faiss_index.ntotal)
            
            start_search = time.time()
            distances, indices = self.faiss_index.search(query_embedding, search_k)
            search_time = time.time() - start_search
            
            logger.debug(f"FAISS search completed in {search_time:.3f}s")
            
            # Convert FAISS results to memories with enhanced scoring
            results = []
            current_time = datetime.now()
            
            for faiss_idx, distance in zip(indices[0], distances[0]):
                if faiss_idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                # Map FAISS index to memory index
                memory_idx = self.index_to_memory_map.get(faiss_idx, faiss_idx)
                
                if 0 <= memory_idx < len(self.memories):
                    memory = self.memories[memory_idx]
                    
                    # Apply metadata filtering
                    memory_type = memory.metadata.get('type', 'history')
                    memory_role = memory.metadata.get('role', 'user')
                    
                    if filter_by_type and memory_type not in filter_by_type:
                        continue
                    if filter_by_role and memory_role not in filter_by_role:
                        continue
                    
                    # Calculate enhanced score with 2025 best practices
                    base_score = float(1.0 / (1.0 + distance))  # Convert L2 distance to similarity
                    importance = memory.metadata.get('importance', 1.0)
                    
                    # Type-based boosting
                    type_multipliers = {
                        'correction': 1.5,
                        'summary': 1.2,
                        'identity': 1.1,
                        'history': 1.0
                    }
                    type_multiplier = type_multipliers.get(memory_type, 1.0)
                    
                    # Age decay with configurable half-life
                    age_days = (current_time - memory.timestamp).total_seconds() / 86400
                    age_factor = 0.5 ** (age_days / age_decay_days) if age_days > 0 else 1.0
                    
                    # Combined score with importance boosting
                    final_score = base_score * (1.0 + importance_boost * importance) * type_multiplier * age_factor
                    
                    if final_score >= score_threshold:
                        memory.relevance_score = final_score
                        results.append(memory)
            
            # Sort by final score and return top k
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            final_results = results[:k]
            
            logger.info(
                f"Optimized search completed",
                extra={
                    "query_length": len(query),
                    "faiss_search_time": search_time,
                    "results_count": len(final_results),
                    "k": k,
                    "cache_stats": self.get_cache_stats() if self.query_cache else None
                }
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during optimized search: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get query cache statistics"""
        if not self.query_cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            **self.query_cache.get_stats()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        stats = {
            "total_memories": len(self.memories),
            "faiss_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "dimension": self.dimension,
            "index_loaded": self.faiss_index is not None,
            "use_hnsw": self.use_hnsw,
            "cache_stats": self.get_cache_stats()
        }
        
        if self.memories:
            timestamps = [m.timestamp for m in self.memories]
            stats.update({
                "oldest_memory": min(timestamps).isoformat(),
                "newest_memory": max(timestamps).isoformat(),
                "average_content_length": sum(len(m.content) for m in self.memories) / len(self.memories)
            })
        
        return stats

def create_optimized_faiss_engine(
    memory_json_path: str,
    index_base_path: str,
    embedding_provider: Optional[EmbeddingProvider] = None,
    **kwargs
) -> OptimizedFAISSMemoryEngine:
    """
    Factory function to create optimized FAISS memory engine
    
    Args:
        memory_json_path: Path to ChatGPT memories JSON
        index_base_path: Base path for FAISS index files
        embedding_provider: Provider for new query embeddings
        **kwargs: Additional parameters for OptimizedFAISSMemoryEngine
        
    Returns:
        Configured and loaded OptimizedFAISSMemoryEngine
    """
    engine = OptimizedFAISSMemoryEngine(
        embedding_provider=embedding_provider,
        index_path=index_base_path,
        **kwargs
    )
    
    success = engine.load_precomputed_index(memory_json_path, index_base_path)
    
    if success:
        logger.info(f"Optimized FAISS engine ready with {len(engine.memories)} memories")
    else:
        logger.error("Failed to load optimized FAISS engine")
    
    return engine