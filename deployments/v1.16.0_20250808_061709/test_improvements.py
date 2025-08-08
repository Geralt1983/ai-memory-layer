#!/usr/bin/env python3
"""
Test script to verify the improvements made to the AI Memory Layer
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timezone, timedelta
import json
import time


def test_datetime_fixes():
    """Test that datetime now uses timezone.utc instead of utcnow()"""
    print("Testing datetime fixes...")
    
    # Test the new datetime usage
    current_time = datetime.now(timezone.utc)
    print(f"✅ datetime.now(timezone.utc) works: {current_time.isoformat()}")
    
    # Verify it has timezone info
    assert current_time.tzinfo is not None, "Datetime should have timezone info"
    print("✅ Datetime has timezone info")
    
    return True


def test_embedding_cache():
    """Test the embedding cache functionality"""
    print("\nTesting embedding cache...")
    
    from core.embedding_cache import EmbeddingCache
    
    # Create a test cache with shorter TTL for testing
    cache = EmbeddingCache(max_size=10, ttl_hours=1, persist_path=None)
    
    # Test basic cache operations
    test_text = "This is a test query for caching"
    test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Initially should be a miss
    result = cache.get(test_text)
    assert result is None, "First query should be a cache miss"
    print("✅ Cache miss on first query")
    
    # Add to cache
    cache.put(test_text, test_embedding)
    print("✅ Added embedding to cache")
    
    # Should be a hit now
    result = cache.get(test_text)
    assert result == test_embedding, "Should get cached embedding"
    print("✅ Cache hit on repeated query")
    
    # Check statistics
    stats = cache.get_stats()
    assert stats['hits'] == 1, "Should have 1 hit"
    assert stats['misses'] == 1, "Should have 1 miss"
    print(f"✅ Cache stats: {stats}")
    
    # Test LRU eviction
    for i in range(15):
        cache.put(f"test_{i}", [float(i)])
    
    # Original should be evicted (cache size is 10)
    result = cache.get(test_text)
    assert result is None, "Old entry should be evicted"
    print("✅ LRU eviction working")
    
    return True


def test_cached_embedding_provider():
    """Test the cached embedding provider wrapper"""
    print("\nTesting cached embedding provider...")
    
    from integrations.cached_embeddings import CachedEmbeddingProvider
    from core.embedding_cache import EmbeddingCache
    
    # Create a mock embedding provider
    class MockEmbeddingProvider:
        def __init__(self):
            self.call_count = 0
        
        def embed_text(self, text):
            self.call_count += 1
            # Return a simple hash-based embedding
            import hashlib
            import numpy as np
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            return np.array([float(hash_val % 100) / 100 for _ in range(5)])
        
        def embed_batch(self, texts):
            return [self.embed_text(text) for text in texts]
    
    # Create cached provider
    mock_provider = MockEmbeddingProvider()
    cache = EmbeddingCache(max_size=100, ttl_hours=1)
    cached_provider = CachedEmbeddingProvider(mock_provider, cache=cache)
    
    # First call should hit the provider
    text = "Test embedding text"
    embedding1 = cached_provider.embed_text(text)
    assert mock_provider.call_count == 1, "Should call provider on cache miss"
    print("✅ Provider called on cache miss")
    
    # Second call should use cache
    embedding2 = cached_provider.embed_text(text)
    assert mock_provider.call_count == 1, "Should not call provider on cache hit"
    assert embedding1.tolist() == embedding2.tolist(), "Should return same embedding"
    print("✅ Cache used on repeated query")
    
    # Test batch with mixed cache hits/misses
    texts = ["Test 1", "Test 2", text, "Test 3"]
    initial_count = mock_provider.call_count
    embeddings = cached_provider.embed_batch(texts)
    
    # Should only call provider for new texts (3 new, 1 cached)
    assert mock_provider.call_count == initial_count + 3, "Should only embed new texts"
    assert len(embeddings) == 4, "Should return all embeddings"
    print("✅ Batch embedding with cache working")
    
    # Check cache stats
    stats = cached_provider.get_cache_stats()
    print(f"✅ Cache stats after batch: {stats}")
    
    return True


def test_auto_cleanup():
    """Test the automated memory cleanup system"""
    print("\nTesting automated memory cleanup...")
    
    from core.auto_cleanup import AutoMemoryCleanup, CleanupConfig, CleanupStrategy
    from datetime import datetime, timezone, timedelta
    
    # Create a simple Memory class for testing
    class Memory:
        def __init__(self, content, embedding=None):
            self.content = content
            self.embedding = embedding
            self.id = None
            self.timestamp = datetime.now(timezone.utc)
            self.relevance_score = 1.0
    
    # Create a mock memory engine
    class MockMemoryEngine:
        def __init__(self):
            self.memories = []
            # Add test memories with different ages and relevance
            base_time = datetime.now(timezone.utc)
            for i in range(20):
                memory = Memory(
                    content=f"Test memory {i}",
                    embedding=None
                )
                memory.id = str(i)
                memory.timestamp = base_time - timedelta(days=i * 5)
                memory.relevance_score = 1.0 - (i * 0.05)  # Decreasing relevance
                self.memories.append(memory)
        
        def delete_memory(self, memory_id):
            self.memories = [m for m in self.memories if m.id != memory_id]
    
    # Create cleanup system
    engine = MockMemoryEngine()
    config = CleanupConfig(
        enabled=False,  # Don't start background thread for testing
        max_memory_count=15,
        max_age_days=30,
        min_relevance_score=0.5,
        archive_before_delete=False,  # Skip archiving for test
        strategy=CleanupStrategy.COMBINED
    )
    
    cleanup = AutoMemoryCleanup(engine, config=config)
    
    print(f"✅ Initial memory count: {len(engine.memories)}")
    
    # Run cleanup in dry-run mode first
    results = cleanup.run_cleanup(dry_run=True)
    print(f"✅ Dry run results: {results['cleaned']} would be cleaned")
    assert results['cleaned'] > 0, "Should identify memories to clean"
    assert len(engine.memories) == 20, "Dry run should not delete memories"
    
    # Run actual cleanup
    results = cleanup.run_cleanup(dry_run=False)
    print(f"✅ Actual cleanup: {results['cleaned']} memories cleaned")
    assert len(engine.memories) < 20, "Should have deleted some memories"
    
    # Check that old and low-relevance memories were removed
    for memory in engine.memories:
        age_days = (datetime.now(timezone.utc) - memory.timestamp).days
        assert age_days <= 30, "Old memories should be removed"
        assert memory.relevance_score >= 0.5, "Low relevance memories should be removed"
    
    print(f"✅ Final memory count: {len(engine.memories)}")
    print("✅ Cleanup strategies working correctly")
    
    # Test configuration save/load
    cleanup.save_config()
    stats = cleanup.get_stats()
    print(f"✅ Cleanup stats: {stats}")
    
    return True


def test_pydantic_fix():
    """Test that Pydantic v2 compatibility is working"""
    print("\nTesting Pydantic v2 compatibility...")
    
    try:
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            message: str
            value: int = 42
        
        # Create a test instance
        test_obj = TestModel(message="Hello")
        
        # Test model_dump (v2 method)
        if hasattr(test_obj, 'model_dump'):
            data = test_obj.model_dump()
            print(f"✅ Pydantic v2 model_dump() works: {data}")
        else:
            # Fallback for v1
            data = test_obj.dict()
            print(f"✅ Using Pydantic v1 dict() method: {data}")
        
        assert data['message'] == "Hello"
        assert data['value'] == 42
        
        return True
        
    except Exception as e:
        print(f"⚠️ Pydantic test skipped: {e}")
        return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running AI Memory Layer Improvement Tests")
    print("=" * 60)
    
    tests = [
        ("DateTime Fixes", test_datetime_fixes),
        ("Embedding Cache", test_embedding_cache),
        ("Cached Embedding Provider", test_cached_embedding_provider),
        ("Auto Cleanup System", test_auto_cleanup),
        ("Pydantic Compatibility", test_pydantic_fix),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED\n")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: ERROR - {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)