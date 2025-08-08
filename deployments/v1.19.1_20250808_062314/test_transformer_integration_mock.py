#!/usr/bin/env python3
"""
Test script for transformer integration with mock fallback
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_mock_transformer_integration():
    """Test that mock transformer integration works correctly"""
    print("🧪 Testing transformer integration with mock fallback...")
    
    try:
        # Test transformer embeddings (should fallback to mock)
        print("\n📍 Testing transformer embedding provider...")
        from integrations.transformer_embeddings import create_transformer_embedding_provider
        
        embedding_provider = create_transformer_embedding_provider(
            fallback_to_mock=True
        )
        
        # Test embedding
        test_text = "Jeremy has two dogs named Remy and Bailey"
        embedding = embedding_provider.embed_text(test_text)
        
        model_info = embedding_provider.get_model_info()
        print(f"✅ Mock embeddings working: {model_info['type']} with dim {model_info['embedding_dimension']}")
        
        # Test memory synthesis
        print("\n📍 Testing memory synthesis...")
        from core.memory_synthesis import get_memory_synthesizer
        
        class MockMemory:
            def __init__(self, content, relevance_score=1.0):
                self.content = content
                self.relevance_score = relevance_score
        
        test_memories = [
            MockMemory("Jeremy has a friendly Golden Retriever named Remy", 0.9),
            MockMemory("Bailey is Jeremy's second dog, also a Golden Retriever but moody", 0.8),
            MockMemory("Jeremy walks his two dogs every morning", 0.7),
        ]
        
        synthesizer = get_memory_synthesizer()
        query = "How many dogs does Jeremy have?"
        synthesized = synthesizer.synthesize_memories(query, test_memories)
        
        print(f"✅ Memory synthesis working: {len(synthesized)} synthesized memories")
        
        if synthesized:
            best = synthesized[0]
            print(f"   Best synthesis: '{best.content}' (confidence: {best.confidence:.2f})")
        
        # Test semantic similarity calculation
        print("\n📍 Testing semantic similarity...")
        similarity = synthesizer.calculate_semantic_similarity(
            "How many dogs?", 
            "Jeremy has two dogs"
        )
        print(f"✅ Semantic similarity working: {similarity:.3f}")
        
        print(f"\n🎉 All mock transformer tests PASSED!")
        print(f"💡 The system successfully falls back to mock implementations")
        print(f"💡 Install transformers and torch for full BERT functionality:")
        print(f"   pip install transformers torch")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock transformer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mock_transformer_integration()
    sys.exit(0 if success else 1)