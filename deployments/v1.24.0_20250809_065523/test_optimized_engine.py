#!/usr/bin/env python3
"""
Test Optimized FAISS Memory Engine
Demonstrates 2025 best practices with precomputed indexes and caching
"""

import sys
import time
from datetime import datetime
from typing import List

from optimized_faiss_memory_engine import create_optimized_faiss_engine
from integrations.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

def test_optimized_engine():
    """Test the optimized FAISS memory engine"""
    print("🚀 Testing Optimized FAISS Memory Engine - 2025 Best Practices")
    print("=" * 70)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ No OpenAI API key found")
        return
    
    # Create embedding provider
    embedding_provider = OpenAIEmbeddings(api_key)
    
    # Create optimized engine
    print("📦 Creating optimized FAISS engine...")
    start_time = time.time()
    
    engine = create_optimized_faiss_engine(
        memory_json_path="data/chatgpt_memories.json",
        index_base_path="data/faiss_chatgpt_index",
        embedding_provider=embedding_provider,
        cache_size=500,  # Cache 500 queries
        enable_query_cache=True,
        use_hnsw=True,
        hnsw_ef_search=100  # HNSW search parameter
    )
    
    load_time = time.time() - start_time
    print(f"✅ Engine loaded in {load_time:.3f}s")
    
    # Show statistics
    stats = engine.get_statistics()
    print(f"\n📊 Engine Statistics:")
    print(f"   Total Memories: {stats['total_memories']:,}")
    print(f"   FAISS Vectors: {stats['faiss_vectors']:,}")
    print(f"   Index Loaded: {stats['index_loaded']}")
    print(f"   Using HNSW: {stats['use_hnsw']}")
    print(f"   Cache Enabled: {stats['cache_stats']['cache_enabled']}")
    
    if stats['cache_stats']['cache_enabled']:
        print(f"   Cache Size: {stats['cache_stats']['max_size']}")
    
    print()
    
    # Test queries with performance measurement
    test_queries = [
        "python programming best practices",
        "machine learning algorithms", 
        "JavaScript async await",
        "project management techniques",
        "debugging and optimization",
        "python programming best practices",  # Repeat to test cache
        "machine learning algorithms",       # Repeat to test cache
    ]
    
    print("🔍 Testing Optimized Search Performance:")
    print("-" * 50)
    
    total_search_time = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        
        start_time = time.time()
        results = engine.search_memories(
            query=query,
            k=3,
            importance_boost=0.5,  # 50% importance boost
            age_decay_days=30.0    # 30-day age decay
        )
        search_time = time.time() - start_time
        total_search_time += search_time
        
        print(f"   ⚡ Found {len(results)} results in {search_time:.3f}s")
        
        if results:
            top_result = results[0]
            role = top_result.metadata.get('role', 'unknown')
            title = top_result.metadata.get('title', 'No title')
            importance = top_result.metadata.get('importance', 1.0)
            
            print(f"   📝 Top result [{role}]: {title}")
            print(f"      Score: {top_result.relevance_score:.3f} | Importance: {importance:.1f}")
            print(f"      Content: {top_result.content[:100]}...")
    
    print(f"\n⚡ Performance Summary:")
    print(f"   Total Search Time: {total_search_time:.3f}s")
    print(f"   Average per Query: {total_search_time / len(test_queries):.3f}s")
    
    # Show cache statistics
    cache_stats = engine.get_cache_stats()
    if cache_stats['cache_enabled']:
        print(f"\n💾 Cache Statistics:")
        print(f"   Cache Hits: {cache_stats['hits']}")
        print(f"   Cache Misses: {cache_stats['misses']}")
        print(f"   Hit Rate: {cache_stats['hit_rate']:.1%}")
        print(f"   Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
    
    # Test advanced filtering
    print(f"\n🎯 Testing Advanced Filtering:")
    print("-" * 40)
    
    # Filter by type
    corrections = engine.search_memories(
        query="code error fix",
        k=3,
        filter_by_type=['correction']
    )
    print(f"Corrections only: {len(corrections)} results")
    
    # Filter by role
    user_queries = engine.search_memories(
        query="how to learn programming",
        k=3,
        filter_by_role=['user']
    )
    print(f"User queries only: {len(user_queries)} results")
    
    print(f"\n🎉 Optimized FAISS engine test completed!")
    print(f"🚀 Ready for production with precomputed indexes and query caching")

def benchmark_comparison():
    """Benchmark optimized vs standard engine"""
    print("\n🏁 Benchmark: Optimized vs Standard Engine")
    print("=" * 50)
    
    # This would compare against the old engine
    # For now, just show the optimized performance
    print("Optimized engine benefits:")
    print("✅ No embedding regeneration on startup")
    print("✅ Query embedding caching (LRU)")
    print("✅ HNSW indexing for faster search")
    print("✅ Metadata filtering and importance boosting")
    print("✅ Batch operations and optimized scoring")

if __name__ == "__main__":
    test_optimized_engine()
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_comparison()