"""
DualWriteEmbeddings: A/B testing provider for embedding comparison.

This provider allows you to send embeddings to two different providers simultaneously
for testing and comparison purposes. Useful for:
- A/B testing new embedding providers
- Gradual migration between providers
- Performance comparison
- Shadow testing production changes
"""

import logging
import time
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..embeddings_interfaces import EmbeddingProvider


class DualWriteEmbeddings:
    """Dual-write embedding provider for A/B testing.
    
    Sends embedding requests to two providers simultaneously:
    - Primary: Used for actual responses (required)
    - Shadow: Used for testing/comparison (optional, failures ignored)
    
    Usage:
        primary = OpenAIEmbeddings()
        shadow = VoyageEmbeddings()  
        embedder = DualWriteEmbeddings(primary=primary, shadow=shadow)
        
        # Returns primary results, but also sends to shadow for testing
        vectors = embedder.embed(["text1", "text2"])
        
        # Check comparison stats
        stats = embedder.get_stats()
    """
    
    def __init__(
        self,
        primary: EmbeddingProvider,
        shadow: Optional[EmbeddingProvider] = None,
        shadow_percentage: float = 100.0,
        compare_results: bool = True,
        max_workers: int = 2
    ):
        """Initialize dual-write embeddings.
        
        Args:
            primary: Primary embedding provider (required, used for responses)
            shadow: Shadow embedding provider (optional, used for testing)
            shadow_percentage: Percentage of requests to send to shadow (0-100)
            compare_results: Whether to compare and log differences between providers
            max_workers: Max thread pool workers for concurrent requests
        """
        self.primary = primary
        self.shadow = shadow
        self.shadow_percentage = max(0.0, min(100.0, shadow_percentage))
        self.compare_results = compare_results
        self.max_workers = max_workers
        
        # Statistics tracking
        self.stats = {
            "primary_requests": 0,
            "primary_failures": 0,
            "shadow_requests": 0,
            "shadow_failures": 0,
            "comparisons": 0,
            "differences": [],
            "total_time": 0,
            "primary_time": 0,
            "shadow_time": 0,
        }
        
        self.logger = logging.getLogger("dualwrite_embeddings")
        
    def _should_shadow(self) -> bool:
        """Determine if this request should go to shadow provider."""
        if not self.shadow:
            return False
        import random
        return random.random() * 100 < self.shadow_percentage
    
    def _compare_embeddings(self, primary_result: List[List[float]], shadow_result: List[List[float]]) -> Dict[str, Any]:
        """Compare embeddings between providers."""
        if not primary_result or not shadow_result:
            return {"error": "Empty results"}
        
        if len(primary_result) != len(shadow_result):
            return {"error": f"Length mismatch: {len(primary_result)} vs {len(shadow_result)}"}
        
        try:
            import numpy as np
            
            differences = []
            for i, (p_vec, s_vec) in enumerate(zip(primary_result, shadow_result)):
                if len(p_vec) != len(s_vec):
                    differences.append(f"Vector {i}: dimension mismatch {len(p_vec)} vs {len(s_vec)}")
                    continue
                
                # Calculate cosine similarity
                p_arr = np.array(p_vec)
                s_arr = np.array(s_vec)
                
                dot_product = np.dot(p_arr, s_arr)
                norms = np.linalg.norm(p_arr) * np.linalg.norm(s_arr)
                cosine_sim = dot_product / norms if norms > 0 else 0
                
                # Calculate L2 distance
                l2_distance = np.linalg.norm(p_arr - s_arr)
                
                differences.append({
                    "index": i,
                    "cosine_similarity": float(cosine_sim),
                    "l2_distance": float(l2_distance)
                })
            
            return {"differences": differences}
            
        except ImportError:
            # Fallback comparison without numpy
            return {"message": "numpy not available for detailed comparison"}
        except Exception as e:
            return {"error": f"Comparison failed: {e}"}
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using dual-write pattern.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Embedding vectors from primary provider
        """
        if not texts:
            return []
        
        start_time = time.time()
        
        # Always get primary result
        primary_result = None
        primary_error = None
        shadow_result = None
        shadow_error = None
        
        should_shadow = self._should_shadow()
        
        if should_shadow and self.shadow:
            # Use thread pool for concurrent requests
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit both requests
                primary_future = executor.submit(self._embed_with_timing, self.primary, texts, "primary")
                shadow_future = executor.submit(self._embed_with_timing, self.shadow, texts, "shadow")
                
                # Wait for both to complete
                for future in as_completed([primary_future, shadow_future]):
                    result, provider_type, duration = future.result()
                    
                    if provider_type == "primary":
                        if isinstance(result, Exception):
                            primary_error = result
                            self.stats["primary_failures"] += 1
                        else:
                            primary_result = result
                            self.stats["primary_time"] += duration
                        self.stats["primary_requests"] += 1
                        
                    elif provider_type == "shadow":
                        if isinstance(result, Exception):
                            shadow_error = result
                            self.stats["shadow_failures"] += 1
                            self.logger.warning(f"Shadow embedding failed: {result}")
                        else:
                            shadow_result = result
                            self.stats["shadow_time"] += duration
                        self.stats["shadow_requests"] += 1
        else:
            # Primary only
            primary_result, _, duration = self._embed_with_timing(self.primary, texts, "primary")
            if isinstance(primary_result, Exception):
                primary_error = primary_result
                self.stats["primary_failures"] += 1
            else:
                self.stats["primary_time"] += duration
            self.stats["primary_requests"] += 1
        
        # Compare results if both succeeded and comparison is enabled
        if (self.compare_results and primary_result and shadow_result and 
            not isinstance(primary_result, Exception) and not isinstance(shadow_result, Exception)):
            comparison = self._compare_embeddings(primary_result, shadow_result)
            self.stats["differences"].append(comparison)
            self.stats["comparisons"] += 1
            
            # Log significant differences
            if "differences" in comparison:
                avg_similarity = sum(d["cosine_similarity"] for d in comparison["differences"] if isinstance(d, dict)) / len(comparison["differences"])
                if avg_similarity < 0.9:  # Log if average similarity is low
                    self.logger.info(f"Embedding difference detected: avg similarity {avg_similarity:.3f}")
        
        self.stats["total_time"] += time.time() - start_time
        
        # Always return primary result or raise primary error
        if primary_error:
            raise primary_error
        
        return primary_result
    
    def _embed_with_timing(self, provider: EmbeddingProvider, texts: List[str], provider_type: str):
        """Embed with timing and error handling."""
        start_time = time.time()
        try:
            result = provider.embed(texts)
            duration = time.time() - start_time
            return result, provider_type, duration
        except Exception as e:
            duration = time.time() - start_time
            return e, provider_type, duration
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text.
        
        Args:
            text: String to embed
            
        Returns:
            Embedding vector from primary provider or None if failed
        """
        try:
            result = self.embed([text])
            return result[0] if result else None
        except Exception:
            return None
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension from primary provider.
        
        Returns:
            Embedding dimension
        """
        return self.primary.get_embedding_dimension()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dual-write statistics.
        
        Returns:
            Dictionary with request counts, timings, and comparison results
        """
        stats = self.stats.copy()
        
        # Calculate averages
        if stats["primary_requests"] > 0:
            stats["primary_avg_time"] = stats["primary_time"] / stats["primary_requests"]
            stats["primary_success_rate"] = 1.0 - (stats["primary_failures"] / stats["primary_requests"])
        
        if stats["shadow_requests"] > 0:
            stats["shadow_avg_time"] = stats["shadow_time"] / stats["shadow_requests"]
            stats["shadow_success_rate"] = 1.0 - (stats["shadow_failures"] / stats["shadow_requests"])
        
        # Calculate comparison summary
        if stats["comparisons"] > 0:
            similarities = []
            for diff in stats["differences"]:
                if "differences" in diff:
                    for d in diff["differences"]:
                        if isinstance(d, dict) and "cosine_similarity" in d:
                            similarities.append(d["cosine_similarity"])
            
            if similarities:
                stats["avg_cosine_similarity"] = sum(similarities) / len(similarities)
                stats["min_cosine_similarity"] = min(similarities)
        
        return stats
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "primary_requests": 0,
            "primary_failures": 0,
            "shadow_requests": 0,
            "shadow_failures": 0,
            "comparisons": 0,
            "differences": [],
            "total_time": 0,
            "primary_time": 0,
            "shadow_time": 0,
        }
    
    def set_shadow_percentage(self, percentage: float):
        """Update shadow percentage for dynamic testing."""
        self.shadow_percentage = max(0.0, min(100.0, percentage))
        self.logger.info(f"Shadow percentage updated to {self.shadow_percentage}%")