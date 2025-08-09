"""
Enhanced semantic search capabilities using transformer embeddings
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from core.memory_engine import MemoryEngine, Memory
from core.logging_config import get_logger, monitor_performance

@dataclass
class SemanticSearchResult:
    """Enhanced search result with semantic similarity scores"""
    memory: Memory
    vector_similarity: float  # Original vector similarity (e.g., from FAISS)
    semantic_similarity: float  # Transformer-based semantic similarity
    combined_score: float  # Weighted combination of both scores
    rank: int  # Final ranking position

class SemanticSearchEnhancer:
    """
    Enhances memory search with transformer-based semantic similarity
    """
    
    def __init__(self, memory_engine: MemoryEngine, semantic_weight: float = 0.6):
        """
        Initialize semantic search enhancer
        
        Args:
            memory_engine: The MemoryEngine to enhance
            semantic_weight: Weight for semantic similarity (0.0 to 1.0)
                           Higher values favor semantic similarity over vector similarity
        """
        self.memory_engine = memory_engine
        self.semantic_weight = semantic_weight
        self.vector_weight = 1.0 - semantic_weight
        self.logger = get_logger("semantic_search")
        
        # Check if memory engine has transformer embeddings
        self.has_transformer = self._check_transformer_support()
        
        if self.has_transformer:
            self.logger.info(
                f"Initialized semantic search enhancer with weights: "
                f"semantic={semantic_weight:.2f}, vector={self.vector_weight:.2f}"
            )
        else:
            self.logger.warning("Memory engine does not have transformer support, using vector search only")
    
    def _check_transformer_support(self) -> bool:
        """Check if the memory engine supports transformer embeddings"""
        if not self.memory_engine.embedding_provider:
            return False
        
        try:
            # Check if embedding provider has transformer capabilities
            model_info = self.memory_engine.embedding_provider.get_model_info()
            return model_info.get('type') in ['transformer_bert', 'transformer']
        except:
            return False
    
    @monitor_performance("semantic_search")
    def enhanced_search(
        self, 
        query: str, 
        k: int = 10,
        min_semantic_threshold: float = 0.3,
        rerank_top_k: int = None
    ) -> List[SemanticSearchResult]:
        """
        Perform enhanced search combining vector and semantic similarity
        
        Args:
            query: Search query
            k: Number of results to return
            min_semantic_threshold: Minimum semantic similarity to include
            rerank_top_k: Number of top vector results to rerank (if None, uses k*2)
        
        Returns:
            List of enhanced search results
        """
        if not self.has_transformer:
            # Fallback to regular vector search
            return self._fallback_vector_search(query, k)
        
        # Determine how many results to get for reranking
        if rerank_top_k is None:
            rerank_top_k = min(k * 2, len(self.memory_engine.memories))
        
        self.logger.debug(f"Enhanced search: query='{query[:50]}...', k={k}, rerank_top_k={rerank_top_k}")
        
        # Step 1: Get initial results from vector search
        initial_results = self.memory_engine.search_memories(query, k=rerank_top_k)
        
        if not initial_results:
            return []
        
        # Step 2: Calculate semantic similarities for reranking
        enhanced_results = []
        query_embedding = self.memory_engine.embedding_provider.embed_text(query)
        
        for i, memory in enumerate(initial_results):
            try:
                # Get vector similarity (from relevance_score if available)
                vector_sim = getattr(memory, 'relevance_score', 1.0 - (i / len(initial_results)))
                
                # Calculate semantic similarity using cosine similarity
                semantic_sim = self._calculate_semantic_similarity(
                    query_embedding, 
                    memory.embedding if memory.embedding is not None else 
                    self.memory_engine.embedding_provider.embed_text(memory.content)
                )
                
                # Only include if semantic similarity meets threshold
                if semantic_sim >= min_semantic_threshold:
                    # Combine scores using weighted average
                    combined_score = (
                        self.semantic_weight * semantic_sim + 
                        self.vector_weight * vector_sim
                    )
                    
                    enhanced_results.append(SemanticSearchResult(
                        memory=memory,
                        vector_similarity=vector_sim,
                        semantic_similarity=semantic_sim,
                        combined_score=combined_score,
                        rank=0  # Will be set after sorting
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Failed to calculate semantic similarity for memory: {e}")
                # Include with vector score only
                enhanced_results.append(SemanticSearchResult(
                    memory=memory,
                    vector_similarity=vector_sim,
                    semantic_similarity=0.0,
                    combined_score=vector_sim * self.vector_weight,
                    rank=0
                ))
        
        # Step 3: Sort by combined score and assign ranks
        enhanced_results.sort(key=lambda x: x.combined_score, reverse=True)
        for i, result in enumerate(enhanced_results):
            result.rank = i + 1
        
        # Step 4: Return top k results
        final_results = enhanced_results[:k]
        
        self.logger.info(
            f"Enhanced search completed: {len(initial_results)} initial → "
            f"{len(enhanced_results)} reranked → {len(final_results)} final results"
        )
        
        return final_results
    
    def _calculate_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # Convert to numpy arrays if needed
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            # Normalize to [0, 1] range
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            self.logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _fallback_vector_search(self, query: str, k: int) -> List[SemanticSearchResult]:
        """Fallback to vector-only search when transformers not available"""
        results = self.memory_engine.search_memories(query, k=k)
        
        enhanced_results = []
        for i, memory in enumerate(results):
            vector_sim = getattr(memory, 'relevance_score', 1.0 - (i / len(results)))
            
            enhanced_results.append(SemanticSearchResult(
                memory=memory,
                vector_similarity=vector_sim,
                semantic_similarity=0.0,  # Not available
                combined_score=vector_sim,
                rank=i + 1
            ))
        
        return enhanced_results
    
    def search_with_context_expansion(
        self, 
        query: str, 
        k: int = 10,
        context_expansion: bool = True,
        expansion_terms: Optional[List[str]] = None
    ) -> List[SemanticSearchResult]:
        """
        Advanced search with automatic context expansion using semantic similarity
        
        Args:
            query: Original search query
            k: Number of results to return
            context_expansion: Whether to expand query with related terms
            expansion_terms: Manual expansion terms (if None, auto-generate)
        
        Returns:
            Enhanced search results with context expansion
        """
        expanded_query = query
        
        if context_expansion and self.has_transformer:
            try:
                # Auto-generate expansion terms based on semantic similarity
                if expansion_terms is None:
                    expansion_terms = self._generate_expansion_terms(query)
                
                if expansion_terms:
                    expanded_query = f"{query} {' '.join(expansion_terms)}"
                    self.logger.debug(f"Expanded query: '{query}' → '{expanded_query}'")
                
            except Exception as e:
                self.logger.warning(f"Context expansion failed: {e}")
        
        # Perform enhanced search with expanded query
        return self.enhanced_search(expanded_query, k=k)
    
    def _generate_expansion_terms(self, query: str, max_terms: int = 3) -> List[str]:
        """
        Generate semantically related terms to expand the query
        """
        # This is a simplified implementation
        # In a production system, you might use word embeddings or knowledge graphs
        
        expansion_map = {
            "dogs": ["pets", "animals", "canines", "puppies"],
            "cats": ["pets", "animals", "felines", "kittens"],
            "pets": ["animals", "dogs", "cats", "companions"],
            "how many": ["count", "number", "quantity", "total"],
            "names": ["called", "named", "identity", "title"],
            "personality": ["character", "behavior", "traits", "temperament"],
        }
        
        query_lower = query.lower()
        expansion_terms = []
        
        for term, expansions in expansion_map.items():
            if term in query_lower:
                expansion_terms.extend(expansions[:max_terms])
        
        return list(set(expansion_terms))  # Remove duplicates
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get analytics about search performance and patterns"""
        return {
            "has_transformer_support": self.has_transformer,
            "semantic_weight": self.semantic_weight,
            "vector_weight": self.vector_weight,
            "memory_count": len(self.memory_engine.memories),
            "embedding_provider": type(self.memory_engine.embedding_provider).__name__
        }


def create_semantic_search_enhancer(memory_engine: MemoryEngine, **kwargs) -> SemanticSearchEnhancer:
    """Factory function to create a SemanticSearchEnhancer"""
    return SemanticSearchEnhancer(memory_engine, **kwargs)