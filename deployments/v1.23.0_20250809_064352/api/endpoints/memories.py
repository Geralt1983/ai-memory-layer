"""
Memory Endpoints  
Handles memory search, stats, and management operations
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

@router.get("/memories/stats")
async def memory_stats(memory_engine):
    """Get comprehensive memory statistics"""
    return {
        "total_memories": len(memory_engine.memories),
        "count": len(memory_engine.memories),
        "faiss_vectors": memory_engine.vector_store.index.ntotal if hasattr(memory_engine.vector_store, 'index') else 0,
        "timestamp": datetime.now().isoformat(),
        "system": "chatgpt_memory_system",
        "status": "active"
    }

@router.get("/stats")
async def general_stats(memory_engine):
    """General system statistics"""
    return {
        "total_memories": len(memory_engine.memories),
        "faiss_vectors": memory_engine.vector_store.index.ntotal if hasattr(memory_engine.vector_store, 'index') else 0,
        "system_info": "optimized_chatgpt_loader",
        "data_source": "chatgpt_conversations",
        "ready": True
    }

@router.post("/memories/search")
async def search_memories(request: SearchRequest, memory_engine):
    """Search memories with similarity scoring"""
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

@router.post("/debug/search")
async def debug_search(request: SearchRequest, memory_engine):
    """Debug endpoint to analyze search quality"""
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