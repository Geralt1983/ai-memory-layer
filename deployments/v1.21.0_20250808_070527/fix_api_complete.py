#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import and create the ChatGPT memory engine
from fixed_direct_chatgpt_api import create_chatgpt_memory_engine
print("🚀 Initializing ChatGPT Memory System...")
memory_engine = create_chatgpt_memory_engine()
print(f"✅ Loaded {len(memory_engine.memories)} ChatGPT memories")

# Import FastAPI and create app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

app = FastAPI(title="ChatGPT Memory API", version="2.0")
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

# All endpoints the frontend needs
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "memory_count": len(memory_engine.memories),
        "vector_store_entries": memory_engine.faiss_index.ntotal,
        "engine_type": "direct_chatgpt"
    }

@app.get("/stats")
async def stats():
    return memory_engine.get_stats()

@app.get("/memories/stats")
async def memories_stats():
    return memory_engine.get_stats()

@app.post("/memories/search")
async def search_memories(request: SearchRequest):
    try:
        results = memory_engine.search_memories(request.query, top_k=request.k)
        return {
            "memories": [
                {
                    "content": r.content,
                    "metadata": r.metadata,
                    "relevance_score": r.relevance_score
                }
                for r in results
            ],
            "total_count": len(results)
        }
    except Exception as e:
        return {"memories": [], "total_count": 0, "error": str(e)}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        relevant_memories = memory_engine.search_memories(request.message, top_k=5)
        context = f"Found {len(relevant_memories)} relevant memories."
        return {
            "response": f"I searched {len(memory_engine.memories):,} ChatGPT memories. {context}",
            "context_used": True,
            "memories_searched": len(memory_engine.memories)
        }
    except Exception as e:
        return {"response": f"Error: {str(e)}", "error": str(e)}

@app.get("/")
async def root():
    return {
        "name": "ChatGPT Memory API",
        "memories_loaded": len(memory_engine.memories),
        "status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    print(f"\n🌐 Starting Complete ChatGPT API Server")
    print(f"📊 Serving {len(memory_engine.memories):,} ChatGPT memories")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
