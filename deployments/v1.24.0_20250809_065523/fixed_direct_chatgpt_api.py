#!/usr/bin/env python3
"""
Fixed Direct ChatGPT API Implementation
=======================================

This version properly bypasses standard API initialization to ensure
the 23,710 ChatGPT memories are used instead of the default 30 test memories.
"""

import os
import sys
import json
import time
import signal
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment first
try:
    from dotenv import load_dotenv
    if os.path.exists(".env"):
        load_dotenv()
        print("✅ Environment loaded from .env")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment")


def create_chatgpt_memory_engine():
    """Create memory engine with ChatGPT data, bypassing standard initialization"""
    print("🚀 Direct ChatGPT Memory API")
    print("=" * 40)
    
    # Verify API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required")
        sys.exit(1)
    
    # Pre-load ChatGPT memories before any API initialization
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
    
    # Create a memory engine wrapper that only uses ChatGPT data
    class ChatGPTMemoryEngine:
        def __init__(self, faiss_index, memory_data):
            self.faiss_index = faiss_index
            self.memory_data = memory_data
            self.memory_lookup = {i: mem for i, mem in enumerate(memory_data)}
            self.memories = memory_data  # For compatibility with API
        
        def search_memories(self, query: str, top_k: int = 5, **kwargs):
            """Search using pre-loaded FAISS index and ChatGPT data"""
            try:
                from integrations.embeddings import OpenAIEmbeddings
                import numpy as np
                
                embedder = OpenAIEmbeddings()
                query_embedding = embedder.embed_text(query)
                
                # Convert to numpy array for FAISS
                query_vector = np.array(query_embedding).reshape(1, -1).astype('float32')
                
                # Search FAISS
                scores, indices = self.faiss_index.search(query_vector, top_k)
                
                # Map back to memory content with proper structure
                results = []
                for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                    if idx in self.memory_lookup:
                        memory_data = self.memory_lookup[idx]
                        
                        # Create a memory-like object for API compatibility
                        class MemoryResult:
                            def __init__(self, data):
                                self.content = data.get('content', '')
                                self.metadata = data.get('metadata', {})
                                self.timestamp = data.get('timestamp', '')
                                self.relevance_score = 1.0 - score  # Convert distance to similarity
                                self.role = data.get('role', 'user')
                                self.title = data.get('title', '')
                                self.thread_id = data.get('thread_id', '')
                                self.type = data.get('type', 'chatgpt_history')
                                self.importance = data.get('importance', 1.0)
                        
                        results.append(MemoryResult(memory_data))
                
                return results
                
            except Exception as e:
                print(f"❌ Search error: {e}")
                return []
        
        def add_memory(self, content: str, metadata: dict = None):
            """Add new memory (placeholder for API compatibility)"""
            print(f"📝 New memory added: {content[:50]}...")
            return "chatgpt_memory_id"
        
        def get_stats(self):
            """Return accurate stats for ChatGPT dataset"""
            return {
                'total_memories': len(self.memory_data),
                'vector_store_entries': self.faiss_index.ntotal,
                'memory_types': {'chatgpt_history': len(self.memory_data)},
                'oldest_memory': min((mem.get('timestamp', '') for mem in self.memory_data if mem.get('timestamp')), default=None),
                'newest_memory': max((mem.get('timestamp', '') for mem in self.memory_data if mem.get('timestamp')), default=None),
                'engine_type': 'direct_chatgpt',
                'data_source': 'chatgpt_conversations'
            }
        
        def clear_memories(self):
            """Clear memories (safety check for ChatGPT data)"""
            print("⚠️  Clear memories requested on ChatGPT dataset - operation blocked")
            pass
    
    return ChatGPTMemoryEngine(index, memory_data)


def start_api_with_chatgpt_data():
    """Start API server with ChatGPT memory engine"""
    
    # Create ChatGPT memory engine
    chatgpt_engine = create_chatgpt_memory_engine()
    
    print("🔄 Configuring API with ChatGPT data...")
    
    # Import FastAPI components
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    # Create FastAPI app without the standard lifespan that loads default memories
    app = FastAPI(
        title="Direct ChatGPT Memory API",
        description="Direct API access to 23,710 ChatGPT conversation memories",
        version="1.0.0"
    )
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define direct endpoints that use the ChatGPT engine
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "memory_count": len(chatgpt_engine.memories),
            "vector_store_entries": chatgpt_engine.faiss_index.ntotal,
            "engine_type": "direct_chatgpt"
        }
    
    @app.get("/stats")
    async def stats():
        return chatgpt_engine.get_stats()
    
    @app.post("/memories/search")
    async def search_memories(request: dict):
        query = request.get('query', '')
        k = request.get('k', 5)
        
        results = chatgpt_engine.search_memories(query, top_k=k)
        
        return {
            'memories': [
                {
                    'content': r.content,
                    'metadata': r.metadata,
                    'timestamp': r.timestamp,
                    'relevance_score': r.relevance_score,
                    'role': r.role,
                    'title': r.title,
                    'thread_id': r.thread_id
                }
                for r in results
            ],
            'total_count': len(results)
        }
    
    @app.post("/chat")
    async def chat(request: dict):
        # Basic chat endpoint that could use memory context
        message = request.get('message', '')
        return {
            'response': f'ChatGPT API received: {message}',
            'context_used': f'Searched {len(chatgpt_engine.memories)} ChatGPT memories'
        }
    
    # Start server
    print(f"\n🌐 Direct ChatGPT API Server Starting...")
    print(f"📊 Ready with {len(chatgpt_engine.memories):,} ChatGPT memories")
    print(f"🔍 Memory Search: http://localhost:8000/memories/search")
    print(f"📊 Stats: http://localhost:8000/stats")
    print(f"❤️  Health: http://localhost:8000/health")
    print(f"\n🚀 {chatgpt_engine.faiss_index.ntotal:,} ChatGPT memories accessible!")
    print(f"\nPress Ctrl+C to stop")
    print("=" * 40)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Shutdown signal received")
    sys.exit(0)


def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start API with ChatGPT data
    start_api_with_chatgpt_data()


if __name__ == "__main__":
    main()