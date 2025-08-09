#!/usr/bin/env python3
"""
Optimized loader for cleaned ChatGPT memories
Uses the improved chatgpt_memories_cleaned.json for better search quality
"""

import json
import os
from pathlib import Path
from optimized_memory_loader import OptimizedChatGPTLoader

class CleanedChatGPTLoader(OptimizedChatGPTLoader):
    """Loader for cleaned and merged ChatGPT memories"""
    
    def __init__(self):
        super().__init__(
            faiss_index_path="data/faiss_chatgpt_cleaned",
            memory_json_path="data/chatgpt_memories_cleaned.json"
        )
    
    def get_stats(self):
        """Get statistics about cleaned memories"""
        if not Path(self.memory_json_path).exists():
            return {"error": "Cleaned memories not found. Run rebuild_memories_clean.py first."}
        
        with open(self.memory_json_path, 'r', encoding='utf-8') as f:
            memories = json.load(f)
        
        # Analyze quality
        stats = {
            'total': len(memories),
            'qa_pairs': 0,
            'merged': 0,
            'avg_length': 0,
            'tiny': 0,
            'small': 0,
            'medium': 0,
            'large': 0,
            'huge': 0
        }
        
        total_length = 0
        for mem in memories:
            content = mem.get('content', '')
            length = len(content)
            total_length += length
            
            # Check if Q&A pair
            if content.startswith('Q:') and '\nA:' in content:
                stats['qa_pairs'] += 1
            
            # Check if merged
            if mem.get('metadata', {}).get('merged'):
                stats['merged'] += 1
            
            # Size distribution
            if length < 20:
                stats['tiny'] += 1
            elif length < 40:
                stats['small'] += 1
            elif length < 200:
                stats['medium'] += 1
            elif length < 1000:
                stats['large'] += 1
            else:
                stats['huge'] += 1
        
        stats['avg_length'] = total_length // len(memories) if memories else 0
        
        return stats

def create_cleaned_chatgpt_engine():
    """Create memory engine with cleaned ChatGPT data"""
    try:
        # Check if cleaned data exists
        if not Path("data/chatgpt_memories_cleaned.json").exists():
            print("⚠️ Cleaned memories not found. Using original data.")
            print("   Run: python scripts/rebuild_memories_clean.py")
            # Fall back to original loader
            from optimized_memory_loader import create_optimized_chatgpt_engine
            return create_optimized_chatgpt_engine()
        
        # Use cleaned data
        loader = CleanedChatGPTLoader()
        
        # Get stats
        stats = loader.get_stats()
        print(f"📊 Cleaned Memory Stats:")
        print(f"   Total: {stats['total']:,}")
        print(f"   Q&A Pairs: {stats['qa_pairs']:,}")
        print(f"   Merged: {stats['merged']:,}")
        print(f"   Avg Length: {stats['avg_length']} chars")
        
        return loader.load_complete_system()
        
    except Exception as e:
        print(f"❌ Failed to load cleaned memories: {e}")
        print("   Falling back to original data...")
        from optimized_memory_loader import create_optimized_chatgpt_engine
        return create_optimized_chatgpt_engine()

if __name__ == "__main__":
    print("🧹 Testing Cleaned ChatGPT Memory Loader...")
    engine = create_cleaned_chatgpt_engine()
    
    if engine:
        print(f"\n✅ Successfully loaded {len(engine.memories):,} cleaned memories")
        
        # Test search
        test_query = "python programming"
        print(f"\n🔍 Testing search for: '{test_query}'")
        results = engine.search_memories(test_query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            content = result.content[:200] + "..." if len(result.content) > 200 else result.content
            print(f"Content: {content}")
            print(f"Relevance: {getattr(result, 'relevance_score', 'N/A')}")