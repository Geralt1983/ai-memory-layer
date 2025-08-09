"""
Test MMR (Maximal Marginal Relevance) diversity algorithm
"""

import pytest
import numpy as np
from core.similarity_utils import mmr
from core.context_builder import ContextBuilder
from core.memory_engine import Memory, MemoryEngine
from unittest.mock import Mock, MagicMock


class TestMMRDiversity:
    """Test MMR algorithm for diverse result selection"""
    
    def setup_method(self):
        """Setup test fixtures"""
        np.random.seed(42)  # For reproducible tests
    
    def test_mmr_basic_functionality(self):
        """Test basic MMR functionality with simple vectors"""
        # Create simple test vectors
        query_vec = np.array([1, 0, 0])
        
        # Document vectors - some similar, some different
        doc_vecs = [
            np.array([1, 0, 0]),      # Identical to query
            np.array([0.9, 0.1, 0]),  # Very similar
            np.array([0, 1, 0]),      # Different
            np.array([0, 0, 1]),      # Different
            np.array([0.8, 0.2, 0]),  # Similar
        ]
        
        # Select 3 documents with balanced lambda
        selected = mmr(query_vec, doc_vecs, k=3, lambda_mult=0.5)
        
        # Should return indices
        assert len(selected) == 3
        assert all(isinstance(idx, int) for idx in selected)
        assert all(0 <= idx < len(doc_vecs) for idx in selected)
        
        # First selected should be most similar to query (index 0)
        assert selected[0] == 0
    
    def test_mmr_diversity_selection(self):
        """Test that MMR selects diverse documents"""
        query_vec = np.array([1, 0])
        
        # Create clusters of similar documents
        doc_vecs = [
            # Cluster 1: High similarity to query
            np.array([1, 0]),
            np.array([0.9, 0.1]),
            np.array([0.95, 0.05]),
            
            # Cluster 2: Medium similarity
            np.array([0.5, 0.5]),
            np.array([0.4, 0.6]),
            
            # Cluster 3: Low similarity
            np.array([0, 1]),
            np.array([0.1, 0.9]),
        ]
        
        # Select with high lambda (favor relevance)
        high_relevance = mmr(query_vec, doc_vecs, k=4, lambda_mult=0.9)
        
        # Select with low lambda (favor diversity)
        high_diversity = mmr(query_vec, doc_vecs, k=4, lambda_mult=0.1)
        
        # High relevance should prefer similar documents
        # High diversity should spread across clusters
        assert high_relevance != high_diversity
    
    def test_mmr_edge_cases(self):
        """Test MMR edge cases"""
        query_vec = np.array([1, 0])
        doc_vecs = [np.array([1, 0]), np.array([0, 1])]
        
        # k=0
        result = mmr(query_vec, doc_vecs, k=0)
        assert result == []
        
        # k larger than available documents
        result = mmr(query_vec, doc_vecs, k=10)
        assert len(result) == 2
        
        # Empty doc_vecs
        result = mmr(query_vec, [], k=5)
        assert result == []
        
        # Single document
        result = mmr(query_vec, [doc_vecs[0]], k=5)
        assert result == [0]
    
    def test_mmr_lambda_parameter_effects(self):
        """Test effects of different lambda values"""
        query_vec = np.array([1, 0, 0])
        
        # Create documents with clear relevance and diversity trade-offs
        doc_vecs = [
            np.array([1, 0, 0]),      # Most relevant
            np.array([0.8, 0.2, 0]),  # Relevant but similar to #0
            np.array([0, 1, 0]),      # Diverse but less relevant
            np.array([0, 0, 1]),      # Diverse but less relevant
        ]
        
        # Pure relevance (lambda=1.0)
        relevance_only = mmr(query_vec, doc_vecs, k=3, lambda_mult=1.0)
        
        # Pure diversity (lambda=0.0)
        diversity_only = mmr(query_vec, doc_vecs, k=3, lambda_mult=0.0)
        
        # Balanced (lambda=0.5)
        balanced = mmr(query_vec, doc_vecs, k=3, lambda_mult=0.5)
        
        # Results should be different
        assert relevance_only != diversity_only
        assert balanced != relevance_only
        assert balanced != diversity_only
        
        # All should start with most relevant document
        assert relevance_only[0] == 0
        assert diversity_only[0] == 0
        assert balanced[0] == 0
    
    def test_context_builder_mmr_integration(self):
        """Test MMR integration with ContextBuilder"""
        # Mock memory engine
        mock_engine = Mock(spec=MemoryEngine)
        mock_engine.vector_store = Mock()
        mock_engine.embedding_provider = Mock()
        
        # Create test memories with embeddings
        memories = []
        embeddings = []
        for i in range(10):
            memory = Mock(spec=Memory)
            memory.content = f"Test memory {i}"
            memory.embedding = np.random.rand(128).tolist()
            memories.append(memory)
            embeddings.append(memory.embedding)
        
        # Mock search_memories to return all memories
        mock_engine.search_memories.return_value = memories
        mock_engine._embed.return_value = np.random.rand(128).tolist()
        
        # Create context builder
        builder = ContextBuilder(mock_engine)
        
        # Test retrieve with MMR
        query = "test query"
        k = 5
        
        results = builder.retrieve(query, k=k)
        
        # Should call search_memories with expanded k
        mock_engine.search_memories.assert_called_once_with(query, k * 3)
        
        # Should return at most k results
        assert len(results) <= k
    
    def test_mmr_with_memory_objects(self):
        """Test MMR with actual Memory objects"""
        # Create query vector
        query_vec = np.random.rand(64)
        
        # Create memory objects with embeddings
        memories = []
        doc_vecs = []
        
        for i in range(8):
            # Create some similar and some diverse embeddings
            if i < 3:
                # Similar to query
                embedding = query_vec + np.random.normal(0, 0.1, 64)
            else:
                # More diverse
                embedding = np.random.rand(64)
            
            memory = Mock(spec=Memory)
            memory.content = f"Memory content {i}"
            memory.embedding = embedding.tolist()
            
            memories.append(memory)
            doc_vecs.append(embedding)
        
        # Apply MMR to select diverse memories
        selected_indices = mmr(query_vec, doc_vecs, k=4, lambda_mult=0.6)
        selected_memories = [memories[i] for i in selected_indices]
        
        assert len(selected_memories) == 4
        assert all(hasattr(mem, 'content') for mem in selected_memories)
    
    def test_mmr_numerical_stability(self):
        """Test MMR numerical stability with extreme values"""
        # Test with very small vectors
        query_vec = np.array([1e-10, 1e-10])
        doc_vecs = [
            np.array([1e-10, 0]),
            np.array([0, 1e-10]),
            np.array([1e-9, 1e-9])
        ]
        
        result = mmr(query_vec, doc_vecs, k=2)
        assert len(result) == 2
        
        # Test with large vectors
        query_vec = np.array([1e6, 1e6])
        doc_vecs = [
            np.array([1e6, 0]),
            np.array([0, 1e6]),
            np.array([5e5, 5e5])
        ]
        
        result = mmr(query_vec, doc_vecs, k=2)
        assert len(result) == 2
    
    def test_mmr_deterministic_behavior(self):
        """Test that MMR produces deterministic results"""
        np.random.seed(123)
        
        query_vec = np.random.rand(32)
        doc_vecs = [np.random.rand(32) for _ in range(10)]
        
        # Run MMR multiple times with same input
        result1 = mmr(query_vec, doc_vecs, k=5, lambda_mult=0.7)
        result2 = mmr(query_vec, doc_vecs, k=5, lambda_mult=0.7)
        result3 = mmr(query_vec, doc_vecs, k=5, lambda_mult=0.7)
        
        # Results should be identical
        assert result1 == result2 == result3
    
    def test_mmr_performance_characteristics(self):
        """Test MMR performance with larger datasets"""
        # Create larger dataset
        n_docs = 1000
        dim = 256
        
        query_vec = np.random.rand(dim)
        doc_vecs = [np.random.rand(dim) for _ in range(n_docs)]
        
        # Should complete in reasonable time
        import time
        start = time.time()
        
        result = mmr(query_vec, doc_vecs, k=50, lambda_mult=0.5)
        
        duration = time.time() - start
        
        # Should return correct number of results
        assert len(result) == 50
        
        # Should complete in reasonable time (less than 5 seconds)
        assert duration < 5.0
        
        # Results should be valid indices
        assert all(0 <= idx < n_docs for idx in result)
        assert len(set(result)) == 50  # All unique
    
    def test_mmr_empty_query_handling(self):
        """Test MMR with zero query vector"""
        query_vec = np.zeros(3)
        doc_vecs = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        
        result = mmr(query_vec, doc_vecs, k=2)
        
        # Should still return valid results
        assert len(result) == 2
        assert all(isinstance(idx, int) for idx in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])