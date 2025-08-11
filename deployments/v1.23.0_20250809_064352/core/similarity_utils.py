#!/usr/bin/env python3
"""
Similarity and Relevance Utilities
Improves search quality with scoring thresholds and semantic filtering
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import re


def mmr(query_vec, doc_vecs: List, k: int = 8, lambda_mult: float = 0.5) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) algorithm for diverse document selection
    
    Balances relevance to query with diversity among selected documents to avoid
    near-duplicates in results.
    
    Args:
        query_vec: Query vector  
        doc_vecs: List of document vectors
        k: Number of documents to select
        lambda_mult: Balance between relevance (1.0) and diversity (0.0)
        
    Returns:
        List of selected document indices
    """
    if not doc_vecs or k <= 0:
        return []
    
    selected = []
    candidates = list(range(len(doc_vecs)))
    
    # Calculate similarity to query for all documents
    query_vec = np.array(query_vec)
    doc_vecs_array = np.array(doc_vecs)
    sim_to_query = np.dot(doc_vecs_array, query_vec)
    
    # Select documents using MMR
    while candidates and len(selected) < k:
        if not selected:
            # First selection: highest similarity to query
            idx = int(np.argmax([sim_to_query[i] for i in candidates]))
            selected.append(candidates.pop(idx))
            continue
        
        # Subsequent selections: balance relevance and diversity
        mmr_scores = []
        
        for cand_idx in candidates:
            # Relevance: similarity to query
            relevance = sim_to_query[cand_idx]
            
            # Diversity: max similarity to already selected documents
            selected_vecs = np.array([doc_vecs[i] for i in selected])
            cand_vec = np.array(doc_vecs[cand_idx])
            diversity = np.max(np.dot(selected_vecs, cand_vec))
            
            # MMR score: weighted combination
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            mmr_scores.append(mmr_score)
        
        # Select candidate with highest MMR score
        best_idx = int(np.argmax(mmr_scores))
        selected.append(candidates.pop(best_idx))
    
    return selected

class RelevanceScorer:
    """Enhanced relevance scoring for memory search results"""
    
    def __init__(self, min_score: float = 0.3, min_length: int = 20):
        """
        Initialize relevance scorer
        
        Args:
            min_score: Minimum similarity score threshold (0-1)
            min_length: Minimum content length to be considered relevant
        """
        self.min_score = min_score
        self.min_length = min_length
    
    def filter_results(self, results: List[Any], scores: List[float], query: str) -> List[Tuple[Any, float]]:
        """
        Filter search results by relevance score and semantic quality
        
        Args:
            results: List of memory/result objects
            scores: List of similarity scores (higher = more similar)  
            query: Original search query
            
        Returns:
            List of (result, score) tuples filtered by relevance
        """
        filtered = []
        query_lower = query.lower().strip()
        
        for result, score in zip(results, scores):
            # Skip if score too low
            if score < self.min_score:
                continue
            
            content = getattr(result, 'content', str(result))
            if isinstance(content, dict):
                content = content.get('content', '')
            
            # Skip if content too short
            if len(content) < self.min_length:
                continue
            
            # Skip obvious fragments
            if self._is_fragment(content):
                continue
            
            # Boost score for semantic relevance
            semantic_score = self._calculate_semantic_boost(content, query_lower)
            adjusted_score = min(1.0, score + semantic_score)
            
            filtered.append((result, adjusted_score))
        
        # Sort by adjusted score (descending)
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return filtered
    
    def _is_fragment(self, content: str) -> bool:
        """Check if content appears to be a meaningless fragment"""
        content = content.strip()
        
        # Very short content
        if len(content) < 15:
            return True
        
        # Single word responses
        if len(content.split()) <= 2:
            return True
        
        # Common filler phrases
        filler_patterns = [
            r'^(yeah|yes|no|okay|ok|sure|hmm|uh|um|ah)\.?$',
            r'^(thanks?|thx)\.?$',
            r'^(got it|i see|alright)\.?$',
            r'^[^\w\s]*$',  # Only punctuation/symbols
        ]
        
        for pattern in filler_patterns:
            if re.match(pattern, content.lower()):
                return True
        
        return False
    
    def _calculate_semantic_boost(self, content: str, query: str) -> float:
        """Calculate semantic relevance boost beyond simple similarity"""
        boost = 0.0
        content_lower = content.lower()
        
        # Exact phrase match
        if query in content_lower:
            boost += 0.2
        
        # Query word overlap
        query_words = set(query.split())
        content_words = set(content_lower.split())
        
        if query_words and content_words:
            overlap = len(query_words & content_words) / len(query_words)
            boost += overlap * 0.1
        
        # Content quality indicators
        if len(content) > 100:  # Substantial content
            boost += 0.05
        
        if any(char in content for char in '.!?'):  # Complete sentences
            boost += 0.05
        
        # Context indicators (questions, explanations)
        context_indicators = ['because', 'therefore', 'however', 'for example', 'specifically']
        if any(indicator in content_lower for indicator in context_indicators):
            boost += 0.1
        
        return min(0.3, boost)  # Cap boost at 0.3

class SearchOptimizer:
    """Optimizes search parameters and results quality"""
    
    def __init__(self):
        self.scorer = RelevanceScorer()
    
    def optimize_search(self, query: str, raw_results: List, raw_scores: List, 
                       target_count: int = 5) -> List[Tuple[Any, float]]:
        """
        Optimize search results for quality and relevance
        
        Args:
            query: Search query string
            raw_results: Raw search results from vector store
            raw_scores: Raw similarity scores
            target_count: Target number of results to return
            
        Returns:
            List of optimized (result, score) tuples
        """
        # Filter by relevance
        filtered = self.scorer.filter_results(raw_results, raw_scores, query)
        
        if not filtered:
            # If no results pass filter, return best raw results with warning
            print(f"⚠️ No results passed relevance filter for query: '{query}'")
            return list(zip(raw_results[:target_count], raw_scores[:target_count]))
        
        # Diversify results to avoid near-duplicates
        diverse_results = self._diversify_results(filtered, target_count)
        
        return diverse_results
    
    def _diversify_results(self, filtered_results: List[Tuple[Any, float]], 
                          target_count: int) -> List[Tuple[Any, float]]:
        """Remove near-duplicate results to increase diversity"""
        if len(filtered_results) <= target_count:
            return filtered_results
        
        diverse = []
        used_content = set()
        
        for result, score in filtered_results:
            content = getattr(result, 'content', str(result))
            if isinstance(content, dict):
                content = content.get('content', '')
            
            # Simple deduplication - check for substantial overlap
            content_words = set(content.lower().split())
            is_duplicate = False
            
            for used in used_content:
                used_words = set(used.lower().split())
                if content_words and used_words:
                    overlap = len(content_words & used_words) / len(content_words | used_words)
                    if overlap > 0.7:  # 70% overlap = duplicate
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                diverse.append((result, score))
                used_content.add(content)
                
                if len(diverse) >= target_count:
                    break
        
        return diverse

def create_search_optimized_engine(memory_engine, min_score: float = 0.3):
    """
    Wrap a memory engine with search optimization
    
    Args:
        memory_engine: Original memory engine
        min_score: Minimum relevance score threshold
        
    Returns:
        Enhanced memory engine with optimized search
    """
    optimizer = SearchOptimizer()
    optimizer.scorer.min_score = min_score
    
    # Store original search method
    original_search = memory_engine.search_memories
    
    def optimized_search(query: str, k: int = 10, **kwargs):
        """Enhanced search with relevance filtering"""
        # Get more raw results to filter from
        raw_k = min(k * 3, 50)  # Get 3x results for filtering
        
        # Get raw results
        raw_results = original_search(query, k=raw_k, **kwargs)
        
        if not raw_results:
            return []
        
        # Extract scores (if available)
        raw_scores = []
        for result in raw_results:
            score = getattr(result, 'relevance_score', 0.5)  # Default score
            raw_scores.append(score)
        
        # Optimize results
        optimized = optimizer.optimize_search(query, raw_results, raw_scores, k)
        
        # Return just the results (maintaining original interface)
        return [result for result, score in optimized]
    
    # Replace search method
    memory_engine.search_memories = optimized_search
    
    return memory_engine

if __name__ == "__main__":
    # Test the relevance scorer
    scorer = RelevanceScorer(min_score=0.4, min_length=25)
    
    # Mock results for testing
    class MockResult:
        def __init__(self, content):
            self.content = content
    
    test_results = [
        MockResult("This is a comprehensive answer about Python programming with detailed examples"),
        MockResult("yeah"),
        MockResult("code"),
        MockResult("Python is a programming language that offers great flexibility"),
        MockResult("ok thanks"),
        MockResult("The solution involves using list comprehensions for efficient data processing")
    ]
    
    test_scores = [0.8, 0.2, 0.3, 0.7, 0.1, 0.6]
    test_query = "python programming"
    
    filtered = scorer.filter_results(test_results, test_scores, test_query)
    
    print("🧪 Relevance Filtering Test:")
    print(f"Query: '{test_query}'")
    print(f"Original results: {len(test_results)}")
    print(f"Filtered results: {len(filtered)}")
    
    for i, (result, score) in enumerate(filtered):
        print(f"{i+1}. Score: {score:.3f} | Content: {result.content[:60]}...")