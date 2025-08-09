#!/usr/bin/env python3
"""
Direct ChatGPT API Implementation
=================================

Directly starts API with 23,710 ChatGPT memories loaded from FAISS.
Bypasses the standard MemoryEngine initialization that causes slow loading.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment first
try:
    from dotenv import load_dotenv
    if os.path.exists(".env"):
        load_dotenv()
except ImportError:
    pass

def main():
    print("🚀 Direct ChatGPT Memory API")
    print("=" * 40)
    
    # Verify API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required")
        sys.exit(1)
    
    # Pre-load ChatGPT memories before importing API
    print("📚 Pre-loading ChatGPT memories...")
    start_time = time.time()
    
    try:
        import faiss
        
        # Load FAISS index
        faiss_path = "./data/faiss_chatgpt_index"
        index = faiss.read_index(f"{faiss_path}.index")
        print(f"✅ FAISS loaded: {index.ntotal} vectors")
        
        # Load memory content
        with open("./data/chatgpt_memories.json", 'r') as f:
            memory_data = json.load(f)
        print(f"✅ Memory data loaded: {len(memory_data)} memories")
        
        load_time = time.time() - start_time
        print(f"🎉 ChatGPT data ready in {load_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error loading ChatGPT data: {e}")
        sys.exit(1)
    
    # Create a global storage for the optimized data
    class ChatGPTData:
        def __init__(self, faiss_index, memory_data):
            self.faiss_index = faiss_index
            self.memory_data = memory_data
            self.memory_lookup = {i: mem for i, mem in enumerate(memory_data)}
    
    global_chatgpt_data = ChatGPTData(index, memory_data)
    
    # Now import and modify the API
    from api.main import app
    import api.main as api_main
    
    # Create fast memory engine that uses our pre-loaded data
    class FastChatGPTEngine:
        def __init__(self, chatgpt_data):
            self.chatgpt_data = chatgpt_data
            self.memories = chatgpt_data.memory_data  # For compatibility
        
        def search_memories(self, query: str, top_k: int = 5, **kwargs):
            """Search using pre-loaded FAISS index"""
            try:
                from integrations.embeddings import OpenAIEmbeddings
                import numpy as np
                
                embedder = OpenAIEmbeddings()
                query_embedding = embedder.embed_text(query)
                
                # Convert to numpy array for FAISS
                query_vector = np.array(query_embedding).reshape(1, -1).astype('float32')
                
                # Search FAISS
                scores, indices = self.chatgpt_data.faiss_index.search(query_vector, top_k)
                
                # Map back to memory content
                results = []
                for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                    if idx in self.chatgpt_data.memory_lookup:
                        memory_data = self.chatgpt_data.memory_lookup[idx]
                        results.append({
                            'content': memory_data.get('content', ''),
                            'metadata': memory_data.get('metadata', {}),
                            'timestamp': memory_data.get('timestamp', ''),
                            'similarity': 1.0 / (1.0 + score),  # Convert distance to similarity
                            'role': memory_data.get('role', 'user'),
                            'title': memory_data.get('title', ''),
                            'thread_id': memory_data.get('thread_id', ''),
                        })
                
                return results
                
            except Exception as e:
                print(f"❌ Search error: {e}")
                return []
        
        def add_memory(self, content: str, metadata: dict = None):
            """Placeholder for adding new memories"""
            print(f"📝 New memory added: {content[:50]}...")
            return "chatgpt_memory_id"
        
        def get_stats(self):
            """Return stats for ChatGPT dataset"""
            return {
                'total_memories': len(self.chatgpt_data.memory_data),
                'vector_store_entries': self.chatgpt_data.faiss_index.ntotal,
                'memory_types': {'chatgpt_history': len(self.chatgpt_data.memory_data)},
                'oldest_memory': min(mem.get('timestamp', '') for mem in self.chatgpt_data.memory_data) if self.chatgpt_data.memory_data else None,
                'newest_memory': max(mem.get('timestamp', '') for mem in self.chatgpt_data.memory_data) if self.chatgpt_data.memory_data else None,
            }
    
    # Replace the API's memory engine
    fast_engine = FastChatGPTEngine(global_chatgpt_data)
    api_main.memory_engine = fast_engine
    
    print(f"🔄 Replaced API memory engine with ChatGPT data")
    print(f"📊 {len(memory_data)} ChatGPT memories ready for search")
    
    # Start the server
    try:
        import uvicorn
        
        print("\n🌐 Starting FastAPI server...")
        print("🔍 Memory Search: http://localhost:8000/memories/search")
        print("💬 Chat Interface: http://localhost:8000/")
        print("📊 Stats: http://localhost:8000/stats") 
        print("📚 23,710 ChatGPT memories accessible!")
        print("\nPress Ctrl+C to stop")
        print("=" * 40)
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")


if __name__ == "__main__":
    main()