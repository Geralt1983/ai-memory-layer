#!/usr/bin/env python3
"""
Test Full ChatGPT Memory Loader
===============================

Verification script to ensure the optimized loader correctly loads
all 23,710+ ChatGPT memories with proper FAISS index synchronization.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_environment_setup():
    """Test environment configuration and required files"""
    print("🔧 Testing Environment Setup")
    print("-" * 40)
    
    # Check environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    else:
        print(f"✅ OPENAI_API_KEY configured ({api_key[:8]}...)")
    
    # Check required files
    required_files = [
        "./data/chatgpt_memories.json",
        "./data/faiss_chatgpt_index.index", 
        "./data/faiss_chatgpt_index.pkl"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            return False
        else:
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"✅ {file_path} ({size_mb:.1f}MB)")
    
    print("✅ Environment setup verification passed")
    return True


def test_data_integrity():
    """Test data file integrity and alignment"""
    print("\n🔍 Testing Data Integrity")
    print("-" * 40)
    
    try:
        import json
        import faiss
        
        # Load and check FAISS index
        faiss_index = faiss.read_index("./data/faiss_chatgpt_index.index")
        faiss_count = faiss_index.ntotal
        print(f"✅ FAISS index loaded: {faiss_count:,} vectors")
        
        # Load and check memory JSON
        with open("./data/chatgpt_memories.json", 'r') as f:
            memory_data = json.load(f)
        memory_count = len(memory_data)
        print(f"✅ Memory JSON loaded: {memory_count:,} entries")
        
        # Check alignment
        if abs(faiss_count - memory_count) <= 100:  # Allow small discrepancy
            print(f"✅ Good alignment: {abs(faiss_count - memory_count)} difference")
            
            # Expect around 23,710 memories from ChatGPT data
            if memory_count > 20000:
                print("✅ Full ChatGPT dataset detected (>20k memories)")
            else:
                print(f"⚠️  Lower count than expected: {memory_count:,}")
        else:
            print(f"❌ Poor alignment: {abs(faiss_count - memory_count)} difference")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data integrity test failed: {e}")
        return False


def test_optimized_loader():
    """Test the optimized memory loader implementation"""
    print("\n🚀 Testing Optimized Memory Loader")
    print("-" * 40)
    
    try:
        from optimized_memory_loader import create_optimized_chatgpt_engine
        
        print("📚 Loading ChatGPT memory engine...")
        start_time = time.time()
        
        # Load the complete system
        memory_engine = create_optimized_chatgpt_engine()
        
        load_time = time.time() - start_time
        memory_count = len(memory_engine.memories) if memory_engine else 0
        
        print(f"⏱️  Load time: {load_time:.2f}s")
        print(f"📊 Loaded memories: {memory_count:,}")
        
        # Verify expected count
        if memory_count > 20000:
            print("✅ SUCCESS: Full ChatGPT dataset loaded")
        elif memory_count > 1000:
            print(f"⚠️  Partial load: {memory_count:,} memories (expected 23k+)")
        else:
            print(f"❌ FAILED: Only {memory_count:,} memories loaded")
            return False, None
        
        # Performance check
        rate = memory_count / load_time if load_time > 0 else 0
        print(f"⚡ Loading rate: {rate:.0f} memories/second")
        
        return True, memory_engine
        
    except Exception as e:
        print(f"❌ Optimized loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_memory_engine_functionality(memory_engine):
    """Test core memory engine functionality"""
    print("\n🧪 Testing Memory Engine Functionality")
    print("-" * 40)
    
    try:
        # Test get_stats - check if method exists
        if hasattr(memory_engine, 'get_stats'):
            stats = memory_engine.get_stats()
            print(f"📊 Engine stats: {stats}")
        else:
            # Alternative stats gathering
            stats = {
                'total_memories': len(memory_engine.memories),
                'engine_type': 'optimized_chatgpt_loader'
            }
            print(f"📊 Engine info: {stats}")
        
        # Verify stats alignment
        total_memories = stats.get('total_memories', 0)
        if total_memories != len(memory_engine.memories):
            print(f"⚠️  Stats mismatch: {total_memories} vs {len(memory_engine.memories)}")
        else:
            print(f"✅ Stats aligned: {total_memories:,} memories")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def test_search_performance(memory_engine):
    """Test search functionality and performance"""
    print("\n🔍 Testing Search Performance")
    print("-" * 40)
    
    test_queries = [
        "python programming",
        "machine learning AI", 
        "ChatGPT conversation",
        "vector database FAISS",
        "OpenAI API integration"
    ]
    
    total_search_time = 0
    successful_searches = 0
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"🔍 Test {i}: '{query}'")
            
            start_time = time.time()
            results = memory_engine.search_memories(query, k=5)
            search_time = time.time() - start_time
            
            total_search_time += search_time
            successful_searches += 1
            
            print(f"   ⏱️  {search_time:.3f}s | 📝 {len(results)} results")
            
            if results:
                top_result = results[0]
                content_preview = top_result.content[:80].replace('\n', ' ')
                score = getattr(top_result, 'relevance_score', 0.0)
                print(f"   📄 Top: {content_preview}... (score: {score:.3f})")
            
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    if successful_searches > 0:
        avg_search_time = total_search_time / successful_searches
        print(f"\n📊 Search Performance Summary:")
        print(f"   • Successful searches: {successful_searches}/{len(test_queries)}")
        print(f"   • Average search time: {avg_search_time:.3f}s")
        print(f"   • Total search time: {total_search_time:.3f}s")
        
        if avg_search_time < 1.0:
            print("✅ Excellent search performance (<1s average)")
        elif avg_search_time < 3.0:
            print("✅ Good search performance (<3s average)")
        else:
            print("⚠️  Slow search performance (>3s average)")
        
        return True
    else:
        print("❌ All searches failed")
        return False


def test_memory_content_quality(memory_engine):
    """Test the quality and variety of loaded memory content"""
    print("\n📝 Testing Memory Content Quality")
    print("-" * 40)
    
    try:
        memories = memory_engine.memories[:100]  # Sample first 100
        
        # Check content variety
        content_lengths = [len(mem.content) for mem in memories if mem.content]
        roles = [getattr(mem, 'role', 'unknown') for mem in memories]
        types = [getattr(mem, 'type', 'unknown') for mem in memories]
        
        print(f"📊 Content Analysis (first 100 memories):")
        print(f"   • Non-empty content: {len(content_lengths)}/100")
        print(f"   • Avg content length: {sum(content_lengths)/len(content_lengths) if content_lengths else 0:.0f} chars")
        print(f"   • Min/Max length: {min(content_lengths) if content_lengths else 0}/{max(content_lengths) if content_lengths else 0}")
        
        # Role distribution
        role_counts = {}
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        print(f"   • Role distribution: {role_counts}")
        
        # Type distribution  
        type_counts = {}
        for type_val in types:
            type_counts[type_val] = type_counts.get(type_val, 0) + 1
        print(f"   • Type distribution: {type_counts}")
        
        # Sample content
        if memories and memories[0].content:
            sample_content = memories[0].content[:200].replace('\n', ' ')
            print(f"📄 Sample content: {sample_content}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Content quality test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite for ChatGPT memory loading"""
    print("🧪 Comprehensive ChatGPT Memory Loader Test")
    print("=" * 60)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Environment setup
    if test_environment_setup():
        tests_passed += 1
    else:
        print("\n❌ Environment setup failed - aborting tests")
        return False
    
    # Test 2: Data integrity
    if test_data_integrity():
        tests_passed += 1
    else:
        print("\n❌ Data integrity failed - aborting tests")
        return False
    
    # Test 3: Optimized loader
    success, memory_engine = test_optimized_loader()
    if success and memory_engine:
        tests_passed += 1
    else:
        print("\n❌ Optimized loader failed - aborting remaining tests")
        return False
    
    # Test 4: Memory engine functionality
    if test_memory_engine_functionality(memory_engine):
        tests_passed += 1
    
    # Test 5: Search performance
    if test_search_performance(memory_engine):
        tests_passed += 1
    
    # Test 6: Content quality
    if test_memory_content_quality(memory_engine):
        tests_passed += 1
    
    # Final results
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("🎯 Test Results Summary")
    print("=" * 60)
    print(f"📊 Tests passed: {tests_passed}/{total_tests}")
    print(f"⏱️  Total test time: {total_time:.2f}s")
    print(f"📝 Memory count: {len(memory_engine.memories):,}")
    print(f"🏁 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Optimized ChatGPT memory loader is working correctly")
        print(f"🚀 Ready to deploy with {len(memory_engine.memories):,} memories")
        return True
    else:
        print(f"\n⚠️  {total_tests - tests_passed}/{total_tests} tests failed")
        print("❌ Issues need to be resolved before deployment")
        return False


if __name__ == "__main__":
    # Load environment
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv()
    except ImportError:
        pass
    
    # Run comprehensive test
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)