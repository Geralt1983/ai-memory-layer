"""Tests for human-like recall scoring functionality."""

import pytest
import time
from memory_layer.retrieval.scoring import human_recall_score, rank_memories, ScoredMemory


@pytest.mark.unit
class TestHumanRecallScoring:
    """Test human-like recall scoring functionality."""
    
    def test_basic_scoring(self):
        """Test basic scoring functionality."""
        # Perfect similarity, recent timestamp
        now = time.time()
        recent_time = now - 3600  # 1 hour ago
        
        score = human_recall_score(
            raw_similarity=1.0,
            timestamp=recent_time,
            now=now,
            tags=None
        )
        
        # Should be high score for recent + high similarity
        assert 0.8 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_temporal_decay(self):
        """Test that older memories get lower scores."""
        now = time.time()
        raw_similarity = 0.8
        
        # Very recent (< 1 hour)
        recent = now - 1800  # 30 minutes
        score_recent = human_recall_score(raw_similarity, recent, now)
        
        # Old (> 1 week) 
        old = now - (7 * 24 * 3600 + 3600)  # More than 1 week
        score_old = human_recall_score(raw_similarity, old, now)
        
        # Recent should score higher
        assert score_recent > score_old
        assert score_recent >= 0.8  # Should maintain high similarity
        assert score_old <= 0.4     # Should be significantly reduced
    
    def test_recency_bands(self):
        """Test different recency time bands."""
        now = time.time()
        raw_similarity = 1.0
        
        # Test each time band
        very_recent = now - 1800      # 30 minutes (< 1 hour)
        recent = now - 7200          # 2 hours (< 24 hours)
        week_old = now - (3 * 24 * 3600)  # 3 days (< 1 week)
        very_old = now - (30 * 24 * 3600)  # 1 month (> 1 week)
        
        score_very_recent = human_recall_score(raw_similarity, very_recent, now)
        score_recent = human_recall_score(raw_similarity, recent, now) 
        score_week_old = human_recall_score(raw_similarity, week_old, now)
        score_very_old = human_recall_score(raw_similarity, very_old, now)
        
        # Should follow decay pattern
        assert score_very_recent >= score_recent
        assert score_recent >= score_week_old  
        assert score_week_old >= score_very_old
        
        # Check approximate expected values
        assert score_very_recent == 1.0  # Perfect recency multiplier
        assert 0.85 <= score_recent <= 0.95  # ~0.9 multiplier
        assert 0.65 <= score_week_old <= 0.75  # ~0.7 multiplier
        assert 0.45 <= score_very_old <= 0.55  # ~0.5 multiplier
    
    def test_salience_boost(self):
        """Test that important tags boost scores."""
        now = time.time()
        timestamp = now - 3600  # 1 hour ago
        raw_similarity = 0.8
        
        # Normal tags
        normal_score = human_recall_score(raw_similarity, timestamp, now, ["info", "data"])
        
        # Important tags
        important_score = human_recall_score(raw_similarity, timestamp, now, ["error", "critical"])
        urgent_score = human_recall_score(raw_similarity, timestamp, now, ["urgent"])
        decision_score = human_recall_score(raw_similarity, timestamp, now, ["decision"])
        milestone_score = human_recall_score(raw_similarity, timestamp, now, ["milestone"])
        
        # Important tags should boost score
        assert important_score > normal_score
        assert urgent_score > normal_score
        assert decision_score > normal_score
        assert milestone_score > normal_score
        
        # Should be roughly 20% boost (1.2x multiplier)
        expected_boost = normal_score * 1.2
        assert abs(important_score - expected_boost) < 0.01
    
    def test_mixed_tags(self):
        """Test behavior with mixed important and normal tags."""
        now = time.time()
        timestamp = now - 3600
        raw_similarity = 0.8
        
        # Mix of important and normal tags
        mixed_score = human_recall_score(
            raw_similarity, timestamp, now, 
            ["info", "error", "data", "normal"]
        )
        
        normal_score = human_recall_score(raw_similarity, timestamp, now, ["info", "data"])
        
        # Should get boost from having any important tag
        assert mixed_score > normal_score
    
    def test_case_insensitive_tags(self):
        """Test that tag matching is case insensitive."""
        now = time.time()
        timestamp = now - 3600
        raw_similarity = 0.8
        
        lower_score = human_recall_score(raw_similarity, timestamp, now, ["error"])
        upper_score = human_recall_score(raw_similarity, timestamp, now, ["ERROR"])
        mixed_score = human_recall_score(raw_similarity, timestamp, now, ["Error"])
        
        # All should give same boost
        assert lower_score == upper_score == mixed_score
    
    def test_score_bounds(self):
        """Test that scores are properly bounded."""
        now = time.time()
        
        # Even with perfect similarity and salience boost, should not exceed 1.0
        perfect_score = human_recall_score(
            raw_similarity=1.0,
            timestamp=now - 1800,  # Very recent
            now=now,
            tags=["critical", "error", "urgent"]  # Multiple important tags
        )
        
        assert perfect_score <= 1.0
        
        # Very low similarity should still give low score
        low_score = human_recall_score(
            raw_similarity=0.1,
            timestamp=now - 1800,
            now=now,
            tags=["critical"]
        )
        
        assert low_score <= 0.15  # Even with boosts, should remain low
    
    def test_default_now_parameter(self):
        """Test that default 'now' parameter works."""
        current_time = time.time()
        recent_timestamp = current_time - 1800  # 30 minutes ago
        
        # Should work without explicit 'now' parameter
        score = human_recall_score(raw_similarity=0.8, timestamp=recent_timestamp)
        
        assert 0.6 <= score <= 1.0
        assert isinstance(score, float)


@pytest.mark.unit
class TestMemoryRanking:
    """Test memory ranking functionality."""
    
    def test_basic_ranking(self):
        """Test basic memory ranking."""
        now = time.time()
        
        # Create test memories with different similarities and timestamps
        memories = [
            ("low similarity recent", [0.1, 0.2, 0.3], now - 1800, {}),
            ("high similarity old", [0.9, 0.8, 0.7], now - (7 * 24 * 3600), {}),
            ("medium similarity medium", [0.6, 0.5, 0.4], now - (24 * 3600), {}),
        ]
        
        query_embedding = [0.8, 0.7, 0.6]
        
        results = rank_memories(memories, query_embedding, k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, ScoredMemory) for r in results)
        
        # Should be sorted by human_score (descending)
        for i in range(len(results) - 1):
            assert results[i].human_score >= results[i + 1].human_score
    
    def test_ranking_with_tags(self):
        """Test ranking considers tag importance."""
        now = time.time()
        
        # Two similar memories, but one has important tags
        memories = [
            ("normal memory", [0.8, 0.7, 0.6], now - 3600, {"tags": ["info"]}),
            ("critical memory", [0.8, 0.7, 0.6], now - 3600, {"tags": ["critical", "error"]}),
        ]
        
        query_embedding = [0.8, 0.7, 0.6]
        
        results = rank_memories(memories, query_embedding, k=2)
        
        # Critical memory should rank higher due to salience boost
        assert results[0].content == "critical memory"
        assert results[0].human_score > results[1].human_score
    
    def test_cosine_similarity_calculation(self):
        """Test that cosine similarity is calculated correctly."""
        now = time.time()
        
        # Perfect match
        memory_embedding = [1.0, 0.0, 0.0]
        query_embedding = [1.0, 0.0, 0.0]
        
        memories = [("perfect match", memory_embedding, now - 1800, {})]
        
        results = rank_memories(memories, query_embedding, k=1)
        
        # Should have perfect similarity
        assert abs(results[0].raw_similarity - 1.0) < 1e-6
    
    def test_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""
        now = time.time()
        
        # Orthogonal vectors should have 0 similarity
        memory_embedding = [1.0, 0.0, 0.0]
        query_embedding = [0.0, 1.0, 0.0]
        
        memories = [("orthogonal", memory_embedding, now - 1800, {})]
        
        results = rank_memories(memories, query_embedding, k=1)
        
        # Should have near-zero similarity
        assert abs(results[0].raw_similarity - 0.0) < 1e-6
    
    def test_k_parameter_limits_results(self):
        """Test that k parameter limits number of results."""
        now = time.time()
        
        # Create many memories
        memories = [
            (f"memory_{i}", [float(i), 0.0, 0.0], now - i * 3600, {})
            for i in range(10)
        ]
        
        query_embedding = [5.0, 0.0, 0.0]  # Should be most similar to memory_5
        
        results = rank_memories(memories, query_embedding, k=3)
        
        # Should return only k=3 results
        assert len(results) == 3
    
    def test_zero_magnitude_handling(self):
        """Test handling of zero-magnitude vectors."""
        now = time.time()
        
        # Zero vector
        memories = [("zero vector", [0.0, 0.0, 0.0], now - 1800, {})]
        query_embedding = [1.0, 0.0, 0.0]
        
        results = rank_memories(memories, query_embedding, k=1)
        
        # Should handle gracefully (cosine sim = 0)
        assert results[0].raw_similarity == 0.0
        assert results[0].human_score >= 0.0
    
    def test_empty_memories_list(self):
        """Test handling of empty memories list."""
        query_embedding = [1.0, 0.0, 0.0]
        
        results = rank_memories([], query_embedding, k=5)
        
        assert results == []
    
    def test_scored_memory_dataclass(self):
        """Test ScoredMemory dataclass properties."""
        now = time.time()
        memory = ScoredMemory(
            content="test content",
            embedding=[1.0, 0.0, 0.0],
            timestamp=now,
            raw_similarity=0.85,
            human_score=0.90,
            metadata={"source": "test"}
        )
        
        assert memory.content == "test content"
        assert memory.embedding == [1.0, 0.0, 0.0]
        assert memory.timestamp == now
        assert memory.raw_similarity == 0.85
        assert memory.human_score == 0.90
        assert memory.metadata["source"] == "test"
    
    def test_ranking_stability(self):
        """Test that ranking is stable for equal scores."""
        now = time.time()
        
        # Create memories with identical similarities and timestamps
        memories = [
            ("memory_a", [1.0, 0.0, 0.0], now - 3600, {}),
            ("memory_b", [1.0, 0.0, 0.0], now - 3600, {}),
            ("memory_c", [1.0, 0.0, 0.0], now - 3600, {}),
        ]
        
        query_embedding = [1.0, 0.0, 0.0]
        
        results = rank_memories(memories, query_embedding, k=3)
        
        # All should have same scores
        scores = [r.human_score for r in results]
        assert all(abs(score - scores[0]) < 1e-6 for score in scores)
        
        # Order should be stable (original order preserved for ties)
        contents = [r.content for r in results]
        assert contents == ["memory_a", "memory_b", "memory_c"]


@pytest.mark.integration
class TestScoringIntegration:
    """Integration tests for scoring with realistic scenarios."""
    
    def test_realistic_memory_ranking(self):
        """Test ranking with realistic memory data."""
        now = time.time()
        
        # Simulate realistic memories with varying properties
        memories = [
            # Recent but low relevance
            ("User logged in successfully", [0.2, 0.1, 0.0], now - 1800, {"tags": ["info"]}),
            
            # Old but high relevance
            ("Critical database connection error occurred", [0.9, 0.8, 0.7], 
             now - (5 * 24 * 3600), {"tags": ["error", "critical"]}),
            
            # Medium age, medium relevance
            ("Updated user preferences settings", [0.6, 0.5, 0.4], 
             now - (12 * 3600), {"tags": ["update"]}),
            
            # Recent and high relevance
            ("System performance degradation detected", [0.8, 0.9, 0.7], 
             now - 3600, {"tags": ["critical", "performance"]}),
        ]
        
        # Query about system issues
        query_embedding = [0.85, 0.9, 0.8]
        
        results = rank_memories(memories, query_embedding, k=4)
        
        # Recent critical issue should rank highest
        assert "performance degradation" in results[0].content
        assert results[0].human_score > 0.8
        
        # All results should be properly structured
        for result in results:
            assert hasattr(result, 'content')
            assert hasattr(result, 'raw_similarity')
            assert hasattr(result, 'human_score')
            assert hasattr(result, 'timestamp')
            assert hasattr(result, 'metadata')
    
    def test_temporal_vs_similarity_tradeoff(self):
        """Test the balance between temporal recency and similarity."""
        now = time.time()
        
        memories = [
            # Very recent but not similar
            ("Recent irrelevant", [0.1, 0.0, 0.0], now - 1800, {}),
            
            # Somewhat old but very similar
            ("Old relevant", [0.95, 0.9, 0.85], now - (2 * 24 * 3600), {}),
        ]
        
        query_embedding = [1.0, 0.9, 0.8]
        
        results = rank_memories(memories, query_embedding, k=2)
        
        # High similarity should generally win over recency for content this old
        # (2 days old isn't extremely old, so similarity should dominate)
        assert results[0].content == "Old relevant"
        assert results[0].raw_similarity > 0.9
        assert results[1].raw_similarity < 0.2