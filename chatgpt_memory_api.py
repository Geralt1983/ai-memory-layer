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
        # Search for relevant memories
        results = memory_engine.search_memories(request.message, k=3)
        
        # Create a helpful response based on the search
        if len(results) > 0:
            # Extract relevant context from memories
            context_pieces = []
            for result in results[:2]:  # Use top 2 most relevant
                context_pieces.append(result.content[:200])
            
            context = " ".join(context_pieces)
            
            # Generate a conversational response
            if "hello" in request.message.lower() or "hi" in request.message.lower():
                response = f"Hello! I can see from your {len(memory_engine.memories):,} ChatGPT conversations that you've discussed many topics. How can I help you today?"
            elif "?" in request.message:
                response = f"Based on your ChatGPT history, I found {len(results)} relevant conversations. Let me help you with that question by referencing what we've discussed before."
            else:
                response = f"I found {len(results)} related conversations in your ChatGPT history. Here's what might be helpful: {context[:300]}..."
        else:
            response = f"I searched through your {len(memory_engine.memories):,} ChatGPT conversations but didn't find directly related content. Feel free to ask me anything - I can help with general questions too!"
        
        return {
            "response": response,
            "relevant_memories": len(results),
            "total_memories": len(memory_engine.memories)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print(f"\n🌐 Starting ChatGPT Memory API with {len(memory_engine.memories):,} memories")
    print("🔗 Available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)