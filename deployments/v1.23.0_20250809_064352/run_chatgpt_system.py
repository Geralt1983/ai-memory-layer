#!/usr/bin/env python3
"""
ChatGPT Memory System Runner
============================

Production runner that loads and serves all 23,710 ChatGPT conversation memories
using the optimized memory loader with FastAPI.
"""

import os
import sys
import time
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Shutting down ChatGPT Memory System...")
    sys.exit(0)


def main():
    """Main entry point for ChatGPT Memory System"""
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 ChatGPT Memory System Startup")
    print("=" * 50)
    
    # Load environment
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv()
            print("✅ Environment loaded from .env")
        else:
            print("⚠️  No .env file found")
    except ImportError:
        print("⚠️  python-dotenv not available")
    
    # Verify API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required")
        print("💡 Please set your OpenAI API key in .env file")
        sys.exit(1)
    
    # Check data files
    required_files = [
        "./data/chatgpt_memories.json",
        "./data/faiss_chatgpt_index.index",
        "./data/faiss_chatgpt_index.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"✅ {file_path} ({size_mb:.1f}MB)")
    
    if missing_files:
        print(f"\n❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Run rebuild_faiss_index.py first to create the FAISS index")
        sys.exit(1)
    
    # Load the complete ChatGPT memory system
    print("\n📚 Loading ChatGPT Memory System...")
    try:
        from optimized_memory_loader import create_optimized_chatgpt_engine
        
        start_time = time.time()
        memory_engine = create_optimized_chatgpt_engine()
        load_time = time.time() - start_time
        
        memory_count = len(memory_engine.memories)
        print(f"\n🎯 ChatGPT Memory System Ready!")
        print(f"📊 Loaded: {memory_count:,} memories in {load_time:.2f}s")
        
        if memory_count < 20000:
            print(f"⚠️  Warning: Expected 23,710+ memories, got {memory_count}")
        
    except Exception as e:
        print(f"❌ Failed to load ChatGPT memory system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Start FastAPI server with the loaded memory engine
    print("\n🌐 Starting FastAPI Server...")
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from typing import List, Optional
        
        # Create FastAPI app
        app = FastAPI(
            title="ChatGPT Memory API",
            description=f"API serving {memory_count:,} ChatGPT conversation memories",
            version="2.0.0"
        )
        
        # Add CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request/Response models
        class SearchRequest(BaseModel):
            query: str
            k: Optional[int] = 5
        
        class MemoryResponse(BaseModel):
            content: str
            role: Optional[str] = None
            timestamp: Optional[str] = None
            relevance_score: Optional[float] = None
            title: Optional[str] = None
            thread_id: Optional[str] = None
        
        class SearchResponse(BaseModel):
            memories: List[MemoryResponse]
            total_count: int
            query_time_ms: float
        
        # API Endpoints
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "memory_count": len(memory_engine.memories),
                "system": "chatgpt_memory_api_v2",
                "dataset_size": f"{len(memory_engine.memories):,} ChatGPT conversations"
            }
        
        @app.get("/stats")
        async def stats():
            return {
                "total_memories": len(memory_engine.memories),
                "faiss_vectors": memory_engine.vector_store.index.ntotal if hasattr(memory_engine.vector_store, 'index') else 0,
                "system_info": "optimized_chatgpt_loader",
                "data_source": "chatgpt_conversations",
                "load_time_seconds": load_time,
                "ready": True
            }
        
        @app.post("/memories/search")
        async def search_memories(request: SearchRequest):
            search_start = time.time()
            
            try:
                results = memory_engine.search_memories(request.query, k=request.k)
                search_time = (time.time() - search_start) * 1000  # Convert to ms
                
                # Convert results to response format
                memory_responses = []
                for result in results:
                    memory_responses.append(MemoryResponse(
                        content=result.content,
                        role=getattr(result, 'role', None),
                        timestamp=str(result.timestamp) if result.timestamp else None,
                        relevance_score=getattr(result, 'relevance_score', None),
                        title=getattr(result, 'title', None),
                        thread_id=getattr(result, 'thread_id', None)
                    ))
                
                return SearchResponse(
                    memories=memory_responses,
                    total_count=len(memory_responses),
                    query_time_ms=search_time
                )
                
            except Exception as e:
                return SearchResponse(
                    memories=[],
                    total_count=0,
                    query_time_ms=(time.time() - search_start) * 1000
                )
        
        @app.post("/chat")
        async def chat(request: dict):
            message = request.get("message", "")
            
            # Simple chat response using memory context
            try:
                # Search for relevant memories
                relevant_memories = memory_engine.search_memories(message, k=3)
                
                context_summary = f"Found {len(relevant_memories)} relevant memories from ChatGPT conversations."
                if relevant_memories:
                    context_summary += f" Most relevant: {relevant_memories[0].content[:100]}..."
                
                return {
                    "response": f"I searched through {len(memory_engine.memories):,} ChatGPT memories. {context_summary}",
                    "context_memories": len(relevant_memories),
                    "total_searchable_memories": len(memory_engine.memories)
                }
            except Exception as e:
                return {
                    "response": f"I have access to {len(memory_engine.memories):,} ChatGPT memories but search failed: {str(e)}",
                    "error": str(e)
                }
        
        # Store globally for access
        app.memory_engine = memory_engine
        
        print(f"🎉 FastAPI server ready with {len(memory_engine.memories):,} ChatGPT memories")
        print("🔗 API endpoints:")
        print("   Health: http://localhost:8000/health")
        print("   Stats: http://localhost:8000/stats") 
        print("   Search: http://localhost:8000/memories/search")
        print("   Chat: http://localhost:8000/chat")
        print("\nPress Ctrl+C to stop")
        print("=" * 50)
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()