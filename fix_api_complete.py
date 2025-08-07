#!/usr/bin/env python3
"""
Complete ChatGPT API with Frontend-Compatible Endpoints
========================================================

This provides all endpoints needed by the frontend with 23,710 ChatGPT memories.
"""

import sys
import os
sys.path.append('.')

# Set up environment
from dotenv import load_dotenv
load_dotenv()

# Import the ChatGPT memory engine
from fixed_direct_chatgpt_api import create_chatgpt_memory_engine, app

# Create the memory engine with all ChatGPT data
print("🚀 Initializing ChatGPT Memory System...")
memory_engine = create_chatgpt_memory_engine()
print(f"✅ Loaded {len(memory_engine.memories)} ChatGPT memories")

# Store the engine in app state
app.state.memory_engine = memory_engine

# Import FastAPI components
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import time

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    context_length: Optional[int] = 2000

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

# Core API endpoints that the frontend needs
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "memory_count": len(memory_engine.memories),
        "vector_store_entries": memory_engine.faiss_index.ntotal,
        "engine_type": "direct_chatgpt"
    }

@app.get("/stats")
async def stats():
    """Statistics endpoint"""
    return memory_engine.get_stats()

@app.get("/memories/stats")
async def memories_stats():
    """Memory statistics endpoint (expected by frontend)"""
    return {
        "total_memories": len(memory_engine.memories),
        "vector_store_entries": memory_engine.faiss_index.ntotal,
        "memory_types": {"chatgpt_history": len(memory_engine.memories)},
        "oldest_memory": min((mem.get('timestamp', '') for mem in memory_engine.memory_data if mem.get('timestamp')), default=None),
        "newest_memory": max((mem.get('timestamp', '') for mem in memory_engine.memory_data if mem.get('timestamp')), default=None),
        "engine_type": "direct_chatgpt",
        "data_source": "chatgpt_conversations"
    }

@app.post("/memories/search")
async def search_memories(request: SearchRequest):
    """Search memories endpoint"""
    try:
        results = memory_engine.search_memories(request.query, top_k=request.k)
        return {
            "memories": [
                {
                    "content": r.content,
                    "metadata": r.metadata,
                    "timestamp": str(r.timestamp) if hasattr(r, 'timestamp') else None,
                    "relevance_score": r.relevance_score,
                    "role": getattr(r, 'role', None),
                    "title": getattr(r, 'title', None),
                    "thread_id": getattr(r, 'thread_id', None)
                }
                for r in results
            ],
            "total_count": len(results)
        }
    except Exception as e:
        return {"memories": [], "total_count": 0, "error": str(e)}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with memory context"""
    try:
        # Search for relevant memories
        relevant_memories = memory_engine.search_memories(request.message, top_k=5)
        
        # Build context from memories
        context = ""
        if relevant_memories:
            context = "Based on previous conversations:\n"
            for mem in relevant_memories[:3]:
                context += f"- {mem.content[:200]}...\n"
        
        # For now, return a response that acknowledges the context
        return {
            "response": f"I found {len(relevant_memories)} relevant memories from our {len(memory_engine.memories):,} ChatGPT conversations. {context}",
            "context_used": len(relevant_memories) > 0,
            "memories_searched": len(memory_engine.memories)
        }
    except Exception as e:
        return {
            "response": f"Error processing chat: {str(e)}",
            "error": str(e)
        }

@app.get("/memories")
async def get_memories(limit: int = 10):
    """Get recent memories"""
    recent = sorted(memory_engine.memory_data, 
                   key=lambda x: x.get('timestamp', ''), 
                   reverse=True)[:limit]
    return {
        "memories": recent,
        "total": len(memory_engine.memories)
    }

# Root endpoint for browser access
@app.get("/")
async def root():
    """Root endpoint - return API info"""
    return {
        "name": "ChatGPT Memory API",
        "version": "2.0",
        "memories_loaded": len(memory_engine.memories),
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "memories_stats": "/memories/stats",
            "search": "/memories/search",
            "chat": "/chat"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🌐 Starting Complete ChatGPT API Server")
    print(f"📊 Serving {len(memory_engine.memories):,} ChatGPT memories")
    print("🔗 Endpoints:")
    print("   /health - System health")
    print("   /stats - Statistics")
    print("   /memories/stats - Memory statistics (frontend)")
    print("   /memories/search - Search memories")
    print("   /chat - Chat with context")
    print("\nPress Ctrl+C to stop")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")