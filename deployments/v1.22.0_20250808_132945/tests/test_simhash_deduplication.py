"""
Test SimHash-based deduplication system
"""

import pytest
from core.memory_manager import simhash_64, hamming_distance, Deduper, SimHashDuplicateCleanup
from core.memory_engine import Memory
from datetime import datetime


class TestSimHashDeduplication:
    """Test SimHash deduplication functionality"""
    
    def test_simhash_identical_text(self):
        """Test that identical texts produce identical hashes"""
        text = "This is a test sentence for SimHash"
        hash1 = simhash_64(text)
        hash2 = simhash_64(text)
        
        assert hash1 == hash2
        assert hamming_distance(hash1, hash2) == 0
    
    def test_simhash_similar_text(self):
        """Test that similar texts produce similar hashes"""
        text1 = "This is a test sentence for SimHash"
        text2 = "This is a test sentence for SimHash algorithm"
        
        hash1 = simhash_64(text1)
        hash2 = simhash_64(text2)
        
        distance = hamming_distance(hash1, hash2)
        assert distance < 10  # Should be quite similar
    
    def test_simhash_different_text(self):
        """Test that different texts produce different hashes"""
        text1 = "This is about machine learning"
        text2 = "The weather is sunny today"
        
        hash1 = simhash_64(text1)
        hash2 = simhash_64(text2)
        
        distance = hamming_distance(hash1, hash2)
        assert distance > 20  # Should be quite different
    
    def test_hamming_distance_calculation(self):
        """Test hamming distance calculation"""
        # Test known values
        assert hamming_distance(0b1010, 0b1100) == 2  # Two different bits
        assert hamming_distance(0b1111, 0b0000) == 4  # All different bits
        assert hamming_distance(0b1010, 0b1010) == 0  # Identical
    
    def test_deduper_first_occurrence(self):
        """Test that first occurrence of text is not marked as duplicate"""
        deduper = Deduper(threshold=6)
        
        text = "This is the first occurrence of this text"
        is_duplicate = deduper.seen_near(text)
        
        assert not is_duplicate
    
    def test_deduper_exact_duplicate(self):
        """Test that exact duplicates are detected"""
        deduper = Deduper(threshold=6)
        
        text = "This is a duplicate text"
        
        # First occurrence
        assert not deduper.seen_near(text)
        
        # Second occurrence - should be detected as duplicate
        assert deduper.seen_near(text)
    
    def test_deduper_near_duplicate(self):
        """Test that near-duplicates are detected"""
        deduper = Deduper(threshold=6)
        
        text1 = "Jeremy likes short, precise answers."
        text2 = "Jeremy prefers precise, short answers."
        
        # First occurrence
        assert not deduper.seen_near(text1)
        
        # Near duplicate - should be detected
        assert deduper.seen_near(text2)
    
    def test_deduper_different_text_not_duplicate(self):
        """Test that sufficiently different texts are not marked as duplicates"""
        deduper = Deduper(threshold=6)
        
        text1 = "This is about machine learning algorithms"
        text2 = "The weather forecast shows rain tomorrow"
        
        # First text
        assert not deduper.seen_near(text1)
        
        # Different text - should not be detected as duplicate
        assert not deduper.seen_near(text2)
    
    def test_deduper_threshold_sensitivity(self):
        """Test that threshold affects duplicate detection"""
        text1 = "This is a test message for similarity"
        text2 = "This is a test message for checking similarity"  # Added words
        
        # Strict threshold
        strict_deduper = Deduper(threshold=3)
        assert not strict_deduper.seen_near(text1)
        assert not strict_deduper.seen_near(text2)  # Should not be detected
        
        # Lenient threshold
        lenient_deduper = Deduper(threshold=15)
        assert not lenient_deduper.seen_near(text1)
        assert lenient_deduper.seen_near(text2)  # Should be detected
    
    def test_simhash_cleanup_strategy(self):
        """Test SimHash cleanup strategy integration"""
        cleanup = SimHashDuplicateCleanup(hamming_threshold=6)
        
        # Create test memories
        memory1 = Memory(
            content="This is the original memory content",
            timestamp=datetime.now()
        )
        
        memory2 = Memory(
            content="This is the original memory content with slight modification", 
            timestamp=datetime.now()
        )
        
        memory3 = Memory(
            content="Completely different content about weather patterns",
            timestamp=datetime.now()
        )
        
        # First memory should not be cleaned up
        assert not cleanup.should_cleanup(memory1, {})
        
        # Similar memory should be cleaned up
        assert cleanup.should_cleanup(memory2, {})
        
        # Different memory should not be cleaned up  
        assert not cleanup.should_cleanup(memory3, {})
    
    def test_simhash_cleanup_priority(self):
        """Test that cleanup priority is based on content length"""
        cleanup = SimHashDuplicateCleanup()
        
        short_memory = Memory(content="Short", timestamp=datetime.now())
        long_memory = Memory(
            content="This is a much longer memory content that should have lower cleanup priority",
            timestamp=datetime.now()
        )
        
        short_priority = cleanup.get_priority(short_memory)
        long_priority = cleanup.get_priority(long_memory)
        
        # Shorter content should have higher priority for cleanup
        assert short_priority > long_priority
    
    def test_bucketing_system(self):
        """Test that bucketing system works for efficient duplicate detection"""
        deduper = Deduper(threshold=6)
        
        # Add many different texts
        texts = [
            "Machine learning is fascinating",
            "Deep learning uses neural networks", 
            "Artificial intelligence will change the world",
            "Natural language processing is complex",
            "Computer vision recognizes patterns"
        ]
        
        for text in texts:
            assert not deduper.seen_near(text)
        
        # Check that buckets were created
        assert len(deduper.buckets) > 0
        
        # Add similar text to first one
        similar_text = "Machine learning is really fascinating"
        assert deduper.seen_near(similar_text)
    
    def test_case_insensitive_hashing(self):
        """Test that SimHash is case insensitive"""
        text1 = "This Is A Test"
        text2 = "this is a test"
        text3 = "THIS IS A TEST"
        
        hash1 = simhash_64(text1)
        hash2 = simhash_64(text2)
        hash3 = simhash_64(text3)
        
        # All should produce identical hashes
        assert hash1 == hash2 == hash3
    
    def test_empty_and_edge_cases(self):
        """Test edge cases like empty strings"""
        # Empty strings
        empty_hash = simhash_64("")
        assert isinstance(empty_hash, int)
        
        # Single word
        single_word = simhash_64("test")
        assert isinstance(single_word, int)
        
        # Very long text
        long_text = "word " * 1000
        long_hash = simhash_64(long_text)
        assert isinstance(long_hash, int)
    
    def test_whitespace_normalization(self):
        """Test that different whitespace patterns are normalized"""
        text1 = "word1 word2 word3"
        text2 = "word1   word2    word3"  # Extra spaces
        text3 = "word1\tword2\nword3"     # Different whitespace chars
        
        hash1 = simhash_64(text1)
        hash2 = simhash_64(text2)
        hash3 = simhash_64(text3)
        
        # Should be very similar (difference mainly from split behavior)
        assert hamming_distance(hash1, hash2) <= 2
        assert hamming_distance(hash1, hash3) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])