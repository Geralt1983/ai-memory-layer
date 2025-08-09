"""
Test embedding cache functionality and get_or_embed API
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch
from core.embedding_cache import EmbeddingCache, get_embedding_cache


class TestEmbeddingCache:
    """Test embedding cache functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.temp_dir, "test_cache.json")
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_initialization(self):
        """Test cache initializes correctly"""
        cache = EmbeddingCache(persist_path=self.cache_path)
        
        assert cache.max_size == 1000  # default
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache._cache) == 0
    
    def test_cache_put_and_get(self):
        """Test basic put and get operations"""
        cache = EmbeddingCache(persist_path=self.cache_path)
        
        text = "test text"
        embedding = [0.1, 0.2, 0.3]
        
        # Put embedding
        cache.put(text, embedding)
        
        # Get embedding
        retrieved = cache.get(text)
        
        assert retrieved == embedding
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss behavior"""
        cache = EmbeddingCache(persist_path=self.cache_path)
        
        # Try to get non-existent embedding
        result = cache.get("non-existent text")
        
        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1
    
    def test_get_or_embed_cache_hit(self):
        """Test get_or_embed with cache hit"""
        cache = EmbeddingCache(persist_path=self.cache_path)
        
        text = "cached text"
        cached_embedding = [0.1, 0.2, 0.3]
        
        # Pre-populate cache
        cache.put(text, cached_embedding)
        
        # Mock embed function
        embed_fn = Mock(return_value=[0.9, 0.8, 0.7])
        
        # Call get_or_embed
        result = cache.get_or_embed(text, embed_fn)
        
        # Should return cached value
        assert result == cached_embedding
        
        # Embed function should not be called
        embed_fn.assert_not_called()
        
        # Stats should show hit
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_get_or_embed_cache_miss(self):
        """Test get_or_embed with cache miss"""
        cache = EmbeddingCache(persist_path=self.cache_path)
        
        text = "new text"
        new_embedding = [0.9, 0.8, 0.7]
        
        # Mock embed function
        embed_fn = Mock(return_value=new_embedding)
        
        # Call get_or_embed
        result = cache.get_or_embed(text, embed_fn)
        
        # Should return new embedding
        assert result == new_embedding
        
        # Embed function should be called once
        embed_fn.assert_called_once_with(text)
        
        # Stats should show miss
        assert cache.hits == 0
        assert cache.misses == 1
        
        # Should now be cached
        cached_result = cache.get(text)
        assert cached_result == new_embedding
    
    def test_cache_persistence(self):
        """Test that cache persists to disk"""
        # Create cache and add data
        cache1 = EmbeddingCache(persist_path=self.cache_path)
        cache1.put("test1", [0.1, 0.2])
        cache1.put("test2", [0.3, 0.4])
        
        # Create new cache from same path
        cache2 = EmbeddingCache(persist_path=self.cache_path)
        
        # Should load persisted data
        assert cache2.get("test1") == [0.1, 0.2]
        assert cache2.get("test2") == [0.3, 0.4]
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        # Small cache size for testing
        cache = EmbeddingCache(max_size=3, persist_path=self.cache_path)
        
        # Fill cache
        cache.put("text1", [0.1])
        cache.put("text2", [0.2])
        cache.put("text3", [0.3])
        
        # All should be cached
        assert cache.get("text1") == [0.1]
        assert cache.get("text2") == [0.2]
        assert cache.get("text3") == [0.3]
        
        # Add one more (should evict oldest)
        cache.put("text4", [0.4])
        
        # text1 should be evicted (oldest)
        assert cache.get("text1") is None
        assert cache.get("text2") == [0.2]
        assert cache.get("text3") == [0.3]
        assert cache.get("text4") == [0.4]
    
    def test_cache_lru_access_updates(self):
        """Test that accessing items updates LRU order"""
        cache = EmbeddingCache(max_size=3, persist_path=self.cache_path)
        
        # Fill cache
        cache.put("text1", [0.1])
        cache.put("text2", [0.2])
        cache.put("text3", [0.3])
        
        # Access text1 (should move to end)
        cache.get("text1")
        
        # Add new item (should evict text2, not text1)
        cache.put("text4", [0.4])
        
        # text2 should be evicted, text1 should remain
        assert cache.get("text1") == [0.1]  # Should still be there
        assert cache.get("text2") is None    # Should be evicted
        assert cache.get("text3") == [0.3]
        assert cache.get("text4") == [0.4]
    
    def test_cache_ttl_expiration(self):
        """Test TTL expiration of cache entries"""
        with patch('core.embedding_cache.datetime') as mock_datetime:
            from datetime import datetime, timezone, timedelta
            
            # Mock current time
            base_time = datetime.now(timezone.utc)
            mock_datetime.now.return_value = base_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Create cache with 1 hour TTL
            cache = EmbeddingCache(ttl_hours=1, persist_path=self.cache_path)
            
            # Add entry
            cache.put("test", [0.1, 0.2])
            assert cache.get("test") == [0.1, 0.2]
            
            # Advance time by 2 hours
            mock_datetime.now.return_value = base_time + timedelta(hours=2)
            
            # Entry should be expired
            assert cache.get("test") is None
            assert cache.misses == 1
    
    def test_cache_stats(self):
        """Test cache statistics calculation"""
        cache = EmbeddingCache(persist_path=self.cache_path)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == '0.00%'
        
        # Add some data and access
        cache.put("test1", [0.1])
        cache.put("test2", [0.2])
        
        # Hit and miss
        cache.get("test1")  # hit
        cache.get("test3")  # miss
        
        stats = cache.get_stats()
        assert stats['size'] == 2
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == '50.00%'
    
    def test_cache_clear(self):
        """Test cache clearing"""
        cache = EmbeddingCache(persist_path=self.cache_path)
        
        # Add data
        cache.put("test1", [0.1])
        cache.put("test2", [0.2])
        cache.get("test1")  # Generate stats
        
        # Clear cache
        cache.clear()
        
        # Should be empty
        assert cache.get("test1") is None
        assert cache.get("test2") is None
        
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
    
    def test_cleanup_expired_entries(self):
        """Test cleanup of expired entries"""
        with patch('core.embedding_cache.datetime') as mock_datetime:
            from datetime import datetime, timezone, timedelta
            
            base_time = datetime.now(timezone.utc)
            mock_datetime.now.return_value = base_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            cache = EmbeddingCache(ttl_hours=1, persist_path=self.cache_path)
            
            # Add entries
            cache.put("fresh", [0.1])
            cache.put("old", [0.2])
            
            # Advance time for one entry to expire
            mock_datetime.now.return_value = base_time + timedelta(hours=2)
            
            # Add fresh entry after time advance
            cache.put("newer", [0.3])
            
            # Cleanup expired
            expired_count = cache.cleanup_expired()
            
            # Should have cleaned up 2 old entries
            assert expired_count == 2
            assert cache.get("newer") == [0.3]  # Fresh entry should remain
            assert cache.get("fresh") is None    # Old entries should be gone
            assert cache.get("old") is None
    
    def test_global_cache_singleton(self):
        """Test global cache singleton behavior"""
        # Get global cache instances
        cache1 = get_embedding_cache(persist_path=self.cache_path)
        cache2 = get_embedding_cache(persist_path=self.cache_path)
        
        # Should be the same instance
        assert cache1 is cache2
        
        # Test that they share state
        cache1.put("shared", [0.1])
        assert cache2.get("shared") == [0.1]
    
    def test_hash_collision_resistance(self):
        """Test that different texts with similar content hash differently"""
        cache = EmbeddingCache(persist_path=self.cache_path)
        
        # Similar but different texts
        texts = [
            "The quick brown fox",
            "The quick brown fox jumps",
            "Quick brown fox the",
            "the quick brown fox"  # Different case
        ]
        
        embeddings = [[i] for i in range(len(texts))]
        
        # Cache all
        for text, embedding in zip(texts, embeddings):
            cache.put(text, embedding)
        
        # All should be retrievable separately
        for text, expected_embedding in zip(texts, embeddings):
            retrieved = cache.get(text)
            assert retrieved == expected_embedding
    
    def test_unicode_text_handling(self):
        """Test that cache handles Unicode text correctly"""
        cache = EmbeddingCache(persist_path=self.cache_path)
        
        # Various Unicode texts
        unicode_texts = [
            "Hello 世界",
            "Café naïve résumé",
            "🚀 Rocket science",
            "Ελληνικά",
            "🎯🔥💯"
        ]
        
        # Cache all with different embeddings
        for i, text in enumerate(unicode_texts):
            cache.put(text, [float(i)])
        
        # All should be retrievable
        for i, text in enumerate(unicode_texts):
            retrieved = cache.get(text)
            assert retrieved == [float(i)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])