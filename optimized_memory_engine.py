#!/usr/bin/env python3
"""
Optimized Memory Engine
Implements 2025 FAISS best practices for fast loading of pre-computed embeddings
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from core.memory_engine import Memory, MemoryEngine
from integrations.embeddings import EmbeddingProvider
from storage.vector_store import VectorStore
from core.logging_config import get_logger, monitor_performance
from core.utils import parse_timestamp

logger = get_logger(__name__)

class OptimizedMemoryEngine(MemoryEngine):
    """
    Enhanced MemoryEngine that follows 2025 FAISS best practices:
    - Uses pre-computed embeddings instead of regenerating
    - Implements batch processing for better performance
    - Supports direct FAISS index loading
    - Optimized for large-scale datasets (23K+ memories)
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
        persist_path: Optional[str] = None,
        auto_save: bool = False,  # Disable auto-save by default to prevent overwrites
        precomputed_mode: bool = True  # Use pre-computed embeddings
    ):
        """
        Initialize optimized memory engine
        
        Args:
            vector_store: Vector storage implementation (FAISS)
            embedding_provider: Optional - only needed for new queries
            persist_path: Optional - path to memory JSON file
            auto_save: Whether to auto-save memories (disabled to prevent overwrites)
            precomputed_mode: Use pre-computed embeddings instead of generating
        """
        self.precomputed_mode = precomputed_mode
        self.auto_save = auto_save
        
        # Initialize parent class
        super().__init__(vector_store, embedding_provider, persist_path)
        
        logger.info(
            "Optimized MemoryEngine initialized",
            extra={
                "precomputed_mode": precomputed_mode,
                "auto_save": auto_save,
                "vector_store_type": type(vector_store).__name__
            }
        )
    
    @monitor_performance("load_precomputed_memories")
    def load_precomputed_memories(
        self,
        memory_json_path: str,
        verify_faiss_alignment: bool = True
    ) -> int:
        """
        Load memories from JSON without regenerating embeddings
        Uses pre-computed FAISS index for embeddings
        
        Args:
            memory_json_path: Path to memory JSON file
            verify_faiss_alignment: Whether to verify memory-FAISS alignment
            
        Returns:
            Number of memories loaded
        """
        logger.info(f"Loading precomputed memories from {memory_json_path}")
        
        if not os.path.exists(memory_json_path):
            logger.error(f"Memory file not found: {memory_json_path}")
            return 0
        
        try:
            # Load JSON data
            with open(memory_json_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            if not memory_data:
                logger.warning("Memory file is empty")
                return 0
            
            logger.info(f"Loaded {len(memory_data)} memories from JSON")
            
            # Convert to Memory objects (batch processing)
            memories = []
            batch_size = 1000
            
            for i in range(0, len(memory_data), batch_size):
                batch = memory_data[i:i + batch_size]
                batch_memories = self._process_memory_batch(batch, i)
                memories.extend(batch_memories)
                
                logger.info(f"Processed batch {i // batch_size + 1}/{(len(memory_data) + batch_size - 1) // batch_size}")
            
            # Set memories directly (skip embedding generation)
            self.memories = memories
            
            # Verify FAISS alignment if requested
            if verify_faiss_alignment:
                self._verify_faiss_alignment()
            
            logger.info(f"Successfully loaded {len(memories)} precomputed memories")
            return len(memories)
            
        except Exception as e:
            logger.error(f"Error loading precomputed memories: {e}")
            return 0
    
    def _process_memory_batch(self, batch: List[Dict], start_index: int) -> List[Memory]:
        """Process a batch of memory data efficiently"""
        memories = []
        
        for i, mem_data in enumerate(batch):
            try:
                # Parse timestamp
                timestamp = parse_timestamp(mem_data.get('timestamp'))
                
                # Create Memory object without embedding
                memory = Memory(
                    content=mem_data.get('content', ''),
                    embedding=None,  # Will use FAISS index for searches
                    metadata=mem_data.get('metadata', {}),
                    timestamp=timestamp,
                    relevance_score=0.0,
                    # Enhanced fields
                    role=mem_data.get('role', 'user'),
                    thread_id=mem_data.get('thread_id'),
                    title=mem_data.get('title'),
                    type=mem_data.get('type', 'history'),
                    importance=mem_data.get('importance', 1.0)
                )
                
                memories.append(memory)
                
            except Exception as e:
                logger.error(f"Error processing memory {start_index + i}: {e}")
                continue
        
        return memories
    
    def _verify_faiss_alignment(self):
        """Verify alignment between memories and FAISS index"""
        try:
            if hasattr(self.vector_store, 'index') and hasattr(self.vector_store.index, 'ntotal'):
                faiss_count = self.vector_store.index.ntotal
                memory_count = len(self.memories)
                
                if faiss_count == memory_count:
                    logger.info(f"✅ Perfect alignment: {memory_count} memories = {faiss_count} FAISS vectors")
                else:
                    logger.warning(f"Alignment check: {memory_count} memories vs {faiss_count} FAISS vectors")
                    logger.info("This is normal if memories were filtered during import")
        except Exception as e:
            logger.error(f"Error verifying FAISS alignment: {e}")
    
    @monitor_performance("search_memories_optimized")
    def search_memories(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Memory]:
        """
        Optimized memory search using pre-computed FAISS embeddings
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum relevance score
            
        Returns:
            List of relevant memories with importance weighting
        """
        if not self.memories:
            logger.warning("No memories available for search")
            return []
        
        if not self.embedding_provider:
            logger.error("No embedding provider available for query processing")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_provider.embed_text(query)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search using FAISS (with pre-computed embeddings)
            # Fetch more results for importance re-ranking
            search_k = min(k * 3, len(self.memories))
            
            similar_indices, similarities = self.vector_store.search(
                query_embedding, search_k
            )
            
            # Convert to Memory objects with enhanced scoring
            results = []
            for idx, similarity in zip(similar_indices, similarities):
                if 0 <= idx < len(self.memories):
                    memory = self.memories[idx]
                    
                    # Apply importance weighting (2025 best practice)
                    base_score = float(similarity)
                    importance_weight = memory.importance
                    
                    # Type-based boosting
                    type_multiplier = {
                        'correction': 1.5,
                        'summary': 1.2,
                        'identity': 1.1,
                        'history': 1.0
                    }.get(memory.type, 1.0)
                    
                    # Age decay (30-day half-life)
                    age_days = (datetime.now() - memory.timestamp).days
                    age_factor = 0.5 ** (age_days / 30.0) if age_days > 0 else 1.0
                    
                    # Combined score
                    final_score = base_score * importance_weight * type_multiplier * age_factor
                    
                    if final_score >= score_threshold:
                        memory.relevance_score = final_score
                        results.append(memory)
            
            # Sort by final score and return top k
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            final_results = results[:k]
            
            logger.info(f"Search completed: {len(final_results)} results for '{query[:50]}'")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during optimized search: {e}")
            return []
    
    def save_memories(self, path: Optional[str] = None) -> bool:
        """
        Save memories to JSON file (only if auto_save is enabled)
        Prevents accidental overwrites of precomputed data
        """
        if not self.auto_save:
            logger.info("Auto-save disabled - skipping memory save to prevent overwrite")
            return True
        
        return super().save_memories(path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics for the optimized memory engine"""
        base_stats = super().get_statistics()
        
        # Add optimization-specific stats
        enhanced_stats = {
            **base_stats,
            "precomputed_mode": self.precomputed_mode,
            "auto_save_enabled": self.auto_save,
            "faiss_vector_count": 0
        }
        
        # Get FAISS vector count
        try:
            if hasattr(self.vector_store, 'index') and hasattr(self.vector_store.index, 'ntotal'):
                enhanced_stats["faiss_vector_count"] = self.vector_store.index.ntotal
        except:
            pass
        
        return enhanced_stats

# Factory function for easy creation
def create_optimized_memory_engine(
    memory_json_path: str,
    faiss_index_path: str,
    embedding_provider: Optional[EmbeddingProvider] = None,
    auto_save: bool = False
) -> OptimizedMemoryEngine:
    """
    Factory function to create an optimized memory engine with pre-computed embeddings
    
    Args:
        memory_json_path: Path to ChatGPT memories JSON
        faiss_index_path: Path to FAISS index (without .index extension)
        embedding_provider: Optional embedding provider for new queries
        auto_save: Whether to enable auto-save (disabled by default)
        
    Returns:
        Configured OptimizedMemoryEngine
    """
    from storage.faiss_store import FaissVectorStore
    
    # Create vector store with pre-computed index
    vector_store = FaissVectorStore(
        dimension=1536,  # OpenAI ada-002 dimension
        index_path=faiss_index_path
    )
    
    # Create optimized memory engine
    memory_engine = OptimizedMemoryEngine(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        persist_path=memory_json_path,
        auto_save=auto_save,
        precomputed_mode=True
    )
    
    # Load precomputed memories
    loaded_count = memory_engine.load_precomputed_memories(memory_json_path)
    
    if loaded_count > 0:
        logger.info(f"Optimized memory engine ready with {loaded_count} memories")
    else:
        logger.error("Failed to load memories into optimized engine")
    
    return memory_engine