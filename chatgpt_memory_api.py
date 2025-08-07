#!/usr/bin/env python3
"""
ChatGPT Memory API - Fixed version without JavaScript escaping issues
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv('.env.local')

# Import optimized loader
from optimized_memory_loader import create_optimized_chatgpt_engine

print("🚀 Loading ChatGPT Memory System...")
memory_engine = create_optimized_chatgpt_engine()
print(f"✅ {len(memory_engine.memories):,} memories loaded!")

# Create FastAPI app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="ChatGPT Memory API", version="2.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatRequest(BaseModel):
    message: str

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

# Root route - serve the enhanced chat interface
@app.get("/", response_class=HTMLResponse)
async def root():
    # Read and return the enhanced web interface
    with open('web_interface_enhanced.html', 'r') as f:
        html_content = f.read()
    # Update the API base URL to work with ngrok
    html_content = html_content.replace("const API_BASE = 'http://localhost:8000';", "const API_BASE = window.location.origin;")
    return html_content

# All the API endpoints
@app.get("/health")
async def health():
    return {
        "status": "healthy",
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
        "ready": True
    }

# Fix: Add missing /memories/stats endpoint for frontend
@app.get("/memories/stats")
async def memory_stats():
    from datetime import datetime
    return {
        "total_memories": len(memory_engine.memories),
        "count": len(memory_engine.memories),
        "faiss_vectors": memory_engine.vector_store.index.ntotal if hasattr(memory_engine.vector_store, 'index') else 0,
        "timestamp": datetime.utcnow().isoformat(),
        "system": "chatgpt_memory_system",
        "status": "active"
    }

# Fix: Add missing conversation title generation endpoint
@app.post("/conversations/generate-title") 
async def generate_title(payload: dict):
    try:
        messages = payload.get("messages", [])
        if not messages:
            return {"title": "New Chat"}
            
        # Extract first user message for title
        first_user_msg = None
        for msg in messages:
            if msg.get("sender") == "user":
                first_user_msg = msg.get("content", "")
                break
                
        if first_user_msg:
            # Create a simple title from first message
            title = first_user_msg[:30].strip()
            if len(first_user_msg) > 30:
                title += "..."
            return {"title": title or "New Chat"}
        else:
            return {"title": "New Chat"}
            
    except Exception as e:
        return {"title": "New Chat", "error": str(e)}

@app.post("/memories/search")
async def search_memories(request: SearchRequest):
    try:
        results = memory_engine.search_memories(request.query, k=request.k)
        return {
            "query": request.query,
            "results": [
                {
                    "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                    "relevance_score": getattr(r, 'relevance_score', 0.0),
                    "timestamp": str(r.timestamp) if hasattr(r, 'timestamp') else None
                }
                for r in results
            ],
            "total_count": len(results),
            "searched_memories": len(memory_engine.memories)
        }
    except Exception as e:
        return {"error": str(e), "results": []}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Search for relevant memories with higher k for better selection
        results = memory_engine.search_memories(request.message, k=5)
        
        # Filter and improve memory quality (relaxed filters for better results)
        quality_memories = []
        for result in results:
            # More lenient filtering - include shorter but meaningful content
            if (len(result.content) > 15 and  # Relaxed minimum length
                not result.content.startswith(("Yeah", "Okay", "Sure", "Hmm", "Uh", "Um")) and  # Skip filler
                len(result.content.split()) > 2):  # Relaxed minimum word count
                quality_memories.append(result)
        
        # Create a helpful response based on the search
        if len(quality_memories) > 0:
            # Extract meaningful context from best memories
            context_pieces = []
            for result in quality_memories[:2]:  # Use top 2 quality memories
                # Take more content for better context
                content = result.content[:400] if len(result.content) > 400 else result.content
                context_pieces.append(f"Previous conversation: {content}")
            
            context = "\n\n".join(context_pieces)
            
            # Generate more helpful conversational responses
            if any(word in request.message.lower() for word in ["hello", "hi", "hey"]):
                response = f"Hello! I found relevant context from your {len(memory_engine.memories):,} ChatGPT conversations. Based on your history, you've discussed various topics. How can I help you today?"
                if quality_memories:
                    response += f"\n\nRecent relevant context:\n{context[:300]}..."
            elif "?" in request.message:
                response = f"I found {len(quality_memories)} relevant conversations in your ChatGPT history that might help answer your question.\n\n{context[:500]}..."
            else:
                if len(context) > 100:
                    response = f"From your ChatGPT conversation history, here's relevant context:\n\n{context[:600]}..."
                else:
                    response = f"I found {len(quality_memories)} related conversations. Here's what might be relevant: {context}"
        else:
            # Fallback when no quality memories found
            response = f"I searched through your {len(memory_engine.memories):,} ChatGPT conversations but didn't find high-quality matches for '{request.message}'. The search found {len(results)} potential matches, but they were too fragmentary to be useful. Try rephrasing your query or asking about specific topics you know you've discussed."
        
        return {
            "response": response,
            "relevant_memories": len(quality_memories),
            "total_memories": len(memory_engine.memories),
            "raw_search_results": len(results)
        }
    except Exception as e:
        return {"error": str(e)}

# Debug endpoint to see raw search results
@app.post("/debug/search")
async def debug_search(request: SearchRequest):
    try:
        results = memory_engine.search_memories(request.query, k=request.k or 10)
        
        debug_results = []
        for i, result in enumerate(results):
            debug_results.append({
                "index": i,
                "content_length": len(result.content),
                "content_preview": result.content[:200],
                "full_content": result.content,
                "relevance_score": getattr(result, 'relevance_score', 0.0),
                "passes_quality_filter": (
                    len(result.content) > 50 and
                    not result.content.startswith(("Yeah", "Okay", "Sure", "Hmm")) and
                    len(result.content.split()) > 8
                )
            })
        
        return {
            "query": request.query,
            "total_results": len(results),
            "results": debug_results
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print(f"\n🌐 Starting ChatGPT Memory API with {len(memory_engine.memories):,} memories")
    print("🔗 Available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)