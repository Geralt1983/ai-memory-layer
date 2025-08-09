#!/usr/bin/env python3
"""
Test script for transformer-based neural network integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.logging_config import get_logger
from integrations.transformer_embeddings import create_transformer_embedding_provider
from integrations.transformer_memory_engine import create_transformer_memory_engine
from core.memory_synthesis import get_memory_synthesizer
from core.semantic_search import create_semantic_search_enhancer

def test_transformer_embeddings():
    """Test transformer embedding provider"""
    logger = get_logger("test_transformer_embeddings")
    logger.info("Testing transformer embedding provider...")
    
    try:
        # Create transformer embedding provider
        embedding_provider = create_transformer_embedding_provider(
            model_name='bert-base-uncased',
            fallback_to_mock=True
        )
        
        # Test single text embedding
        test_text = "Jeremy has two dogs named Remy and Bailey"
        embedding = embedding_provider.embed_text(test_text)
        
        logger.info(f"✅ Single embedding test passed - shape: {embedding.shape}")
        
        # Test batch embeddings
        test_texts = [
            "Remy is a friendly Golden Retriever",
            "Bailey is a moody Golden Retriever",
            "How many dogs does Jeremy have?"
        ]
        batch_embeddings = embedding_provider.embed_batch(test_texts)
        
        logger.info(f"✅ Batch embedding test passed - {len(batch_embeddings)} embeddings")
        
        # Test model info
        model_info = embedding_provider.get_model_info()
        logger.info(f"✅ Model info: {model_info}")
        
        return True, embedding_provider
        
    except Exception as e:
        logger.error(f"❌ Transformer embeddings test failed: {e}")
        return False, None

def test_memory_synthesis():
    """Test enhanced memory synthesis with semantic similarity"""
    logger = get_logger("test_memory_synthesis")
    logger.info("Testing enhanced memory synthesis...")
    
    try:
        # Create mock memories for testing
        class MockMemory:
            def __init__(self, content, relevance_score=1.0):
                self.content = content
                self.relevance_score = relevance_score
        
        test_memories = [
            MockMemory("Jeremy has a friendly Golden Retriever named Remy", 0.9),
            MockMemory("Bailey is Jeremy's second dog, also a Golden Retriever but moody", 0.8),
            MockMemory("Jeremy walks his two dogs every morning", 0.7),
            MockMemory("Remy and Bailey love playing in the park", 0.8),
            MockMemory("Jeremy mentioned having pets", 0.6)
        ]
        
        # Test synthesis
        synthesizer = get_memory_synthesizer()
        
        # Test pet count synthesis
        query = "How many dogs does Jeremy have?"
        synthesized = synthesizer.synthesize_memories(query, test_memories)
        
        logger.info(f"✅ Memory synthesis test passed - {len(synthesized)} synthesized memories")
        
        for i, syn_mem in enumerate(synthesized[:3]):  # Show top 3
            logger.info(f"  Synthesis {i+1}: {syn_mem.content[:100]}... (confidence: {syn_mem.confidence:.2f})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory synthesis test failed: {e}")
        return False

def test_transformer_memory_engine():
    """Test transformer-enhanced memory engine"""
    logger = get_logger("test_transformer_memory_engine")
    logger.info("Testing transformer memory engine...")
    
    try:
        # Create transformer memory engine
        memory_engine = create_transformer_memory_engine(
            fallback_to_mock=True,
            persist_path=None  # Don't persist for testing
        )
        
        # Add test memories
        test_memories = [
            "Jeremy has two Golden Retrievers named Remy and Bailey",
            "Remy is very friendly and loves meeting new people",
            "Bailey tends to be moody but is loyal to Jeremy",
            "Jeremy walks his dogs every morning at 7 AM",
            "The dogs love playing fetch in the local park"
        ]
        
        for content in test_memories:
            memory_engine.add_memory(content)
        
        logger.info(f"✅ Added {len(test_memories)} memories to transformer engine")
        
        # Test search
        search_results = memory_engine.search_memories("How many dogs does Jeremy have?", k=3)
        logger.info(f"✅ Search test passed - found {len(search_results)} results")
        
        for i, result in enumerate(search_results):
            logger.info(f"  Result {i+1}: {result.content[:80]}... (score: {getattr(result, 'relevance_score', 0.0):.3f})")
        
        return True, memory_engine
        
    except Exception as e:
        logger.error(f"❌ Transformer memory engine test failed: {e}")
        return False, None

def test_semantic_search():
    """Test semantic search enhancer"""
    logger = get_logger("test_semantic_search")
    logger.info("Testing semantic search enhancer...")
    
    try:
        # Use memory engine from previous test
        success, memory_engine = test_transformer_memory_engine()
        if not success or not memory_engine:
            logger.warning("Skipping semantic search test - memory engine test failed")
            return False
        
        # Create semantic search enhancer
        search_enhancer = create_semantic_search_enhancer(
            memory_engine, 
            semantic_weight=0.7
        )
        
        # Test enhanced search
        query = "Tell me about Jeremy's pets"
        enhanced_results = search_enhancer.enhanced_search(query, k=3)
        
        logger.info(f"✅ Semantic search test passed - {len(enhanced_results)} enhanced results")
        
        for i, result in enumerate(enhanced_results):
            logger.info(f"  Enhanced result {i+1}: {result.memory.content[:80]}...")
            logger.info(f"    Vector sim: {result.vector_similarity:.3f}, Semantic sim: {result.semantic_similarity:.3f}, Combined: {result.combined_score:.3f}")
        
        # Test analytics
        analytics = search_enhancer.get_search_analytics()
        logger.info(f"✅ Search analytics: {analytics}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Semantic search test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    logger = get_logger("comprehensive_test")
    logger.info("🚀 Starting comprehensive transformer integration test...")
    
    results = {}
    
    # Test 1: Transformer embeddings
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Transformer Embeddings")
    logger.info("="*60)
    results['embeddings'] = test_transformer_embeddings()[0]
    
    # Test 2: Memory synthesis
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Enhanced Memory Synthesis")
    logger.info("="*60)
    results['synthesis'] = test_memory_synthesis()
    
    # Test 3: Transformer memory engine
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Transformer Memory Engine")
    logger.info("="*60)
    results['memory_engine'] = test_transformer_memory_engine()[0]
    
    # Test 4: Semantic search
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Semantic Search Enhancement")
    logger.info("="*60)
    results['semantic_search'] = test_semantic_search()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name.upper()}: {status}")
    
    logger.info(f"\nOVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All transformer integration tests PASSED!")
    else:
        logger.warning(f"⚠️ {total - passed} tests FAILED - check logs above")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Test runner crashed: {e}")
        sys.exit(1)