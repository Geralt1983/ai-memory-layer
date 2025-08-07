#!/usr/bin/env python3
"""
Ultra-Fast ChatGPT Memory Loader
================================

Minimal, streamlined loader that pairs FAISS vectors with memory content
in the most efficient way possible for instant API startup.
"""

import json
import os
import time
from typing import Dict, Any, Optional
import faiss

# API components will be imported dynamically


class FastMemoryPatcher:
    """Fast memory system patcher that replaces the global memory engine"""
    
    @staticmethod
    def patch_api_with_chatgpt_data():
        """Replace the API's memory engine with ChatGPT data"""
        
        print("⚡ Fast ChatGPT Memory Patcher")
        print("=" * 40)
        
        start_time = time.time()
        
        # Load FAISS index
        faiss_path = "./data/faiss_chatgpt_index"
        print(f"📥 Loading FAISS index from {faiss_path}")
        
        try:
            index = faiss.read_index(f"{faiss_path}.index")
            print(f"✅ FAISS loaded: {index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error loading FAISS: {e}")
            return False
        
        # Create a minimal vector store wrapper
        class FastVectorStore:
            def __init__(self, faiss_index):
                self.index = faiss_index
                self.dimension = 1536
                
            def search(self, query_vector, top_k=5):
                # Convert query to proper format for FAISS
                import numpy as np
                if hasattr(query_vector, 'reshape'):
                    query_vector = query_vector.reshape(1, -1).astype('float32')
                else:
                    query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
                    
                scores, indices = self.index.search(query_vector, top_k)
                return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        
        # Load memory metadata (streaming for efficiency)
        memory_path = "./data/chatgpt_memories.json"
        print(f"📚 Loading memory metadata from {memory_path}")
        
        try:
            with open(memory_path, 'r') as f:
                memory_data = json.load(f)
            print(f"✅ Memory data loaded: {len(memory_data)} memories")
        except Exception as e:
            print(f"❌ Error loading memories: {e}")
            return False
        
        # Create fast lookup for memory content by index
        memory_lookup = {i: mem for i, mem in enumerate(memory_data)}
        
        # Replace global memory engine with fast version
        global memory_engine
        
        class FastMemoryEngine:
            def __init__(self, vector_store, memory_lookup):
                self.vector_store = vector_store
                self.memory_lookup = memory_lookup
                self.memories = list(memory_lookup.values())  # For compatibility
                
            def search_memories(self, query: str, top_k: int = 5, **kwargs):
                """Fast search using pre-computed embeddings"""
                
                # Get embedding for query (this is the only embedding we need to compute)
                from integrations.embeddings import OpenAIEmbeddings
                embedder = OpenAIEmbeddings()
                
                try:
                    query_embedding = embedder.embed_text(query)
                    
                    # Search FAISS
                    results = self.vector_store.search(query_embedding, top_k)
                    
                    # Map results back to memory content
                    search_results = []
                    for idx, score in results:
                        if idx in self.memory_lookup:
                            memory_data = self.memory_lookup[idx]
                            search_results.append({
                                'content': memory_data.get('content', ''),
                                'metadata': memory_data.get('metadata', {}),
                                'timestamp': memory_data.get('timestamp', ''),
                                'similarity': 1.0 - score,  # Convert distance to similarity
                                'role': memory_data.get('role', 'user'),
                                'title': memory_data.get('title', ''),
                                'thread_id': memory_data.get('thread_id', ''),
                            })
                    
                    return search_results
                    
                except Exception as e:
                    print(f"Search error: {e}")
                    return []
            
            def add_memory(self, content: str, metadata: dict = None):
                """Add new memory (for API compatibility)"""
                # For now, just log - could implement later
                print(f"Adding new memory: {content[:50]}...")
                return "fast_memory_id"
                
            def get_stats(self):
                """Get memory statistics"""
                return {
                    'total_memories': len(self.memory_lookup),
                    'vector_store_entries': self.vector_store.index.ntotal,
                    'memory_types': {'chatgpt_history': len(self.memory_lookup)},
                }
        
        # Create fast vector store and memory engine
        vector_store = FastVectorStore(index)
        fast_engine = FastMemoryEngine(vector_store, memory_lookup)
        
        # Replace global memory engine in API
        import api.main as api_main
        api_main.memory_engine = fast_engine
        
        load_time = time.time() - start_time
        print(f"🎉 Fast patcher completed in {load_time:.2f}s")
        print(f"📊 Ready with {len(memory_data)} ChatGPT memories")
        
        return True


def patch_api_startup():
    """Call this to patch the API with ChatGPT data at startup"""
    return FastMemoryPatcher.patch_api_with_chatgpt_data()


if __name__ == "__main__":
    # Test the fast patcher
    success = patch_api_startup()
    if success:
        print("✅ Fast ChatGPT patcher test successful!")
    else:
        print("❌ Fast patcher test failed")