#!/usr/bin/env python3
"""
Simplified API server that demonstrates core functionality without numpy dependencies
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Import our simple test classes
from simple_test import SimpleMemory, SimpleMemoryEngine

app = FastAPI(title="AI Memory Layer - Simple API", version="1.0.0")

# Global memory engine instance
memory_engine = SimpleMemoryEngine()


# Pydantic models for API
class MemoryCreate(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    content: str
    metadata: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    memory_count: int
    timestamp: str


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        memory_count=memory_engine.get_memory_count(),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/memories", response_model=MemoryResponse)
async def create_memory(memory_data: MemoryCreate):
    """Create a new memory"""
    # Create a simple embedding (mock data since we can't use real embeddings)
    mock_embedding = [hash(memory_data.content) % 100 / 100.0 for _ in range(3)]

    memory = SimpleMemory(
        content=memory_data.content,
        embedding=mock_embedding,
        metadata=memory_data.metadata or {},
        timestamp=datetime.now(),
    )

    memory_engine.add_memory(memory)

    return MemoryResponse(
        content=memory.content,
        metadata=memory.metadata,
        timestamp=memory.timestamp.isoformat(),
    )


@app.get("/memories", response_model=List[MemoryResponse])
async def get_memories(limit: int = 10):
    """Get recent memories"""
    memories = memory_engine.get_recent_memories(limit)

    return [
        MemoryResponse(
            content=memory.content,
            metadata=memory.metadata,
            timestamp=memory.timestamp.isoformat(),
        )
        for memory in memories
    ]


@app.post("/memories/search", response_model=List[MemoryResponse])
async def search_memories(search_data: SearchRequest):
    """Search memories (simplified - just returns recent memories)"""
    # In a real implementation, this would use vector similarity
    # For now, return recent memories filtered by content match
    all_memories = memory_engine.get_recent_memories(100)

    # Simple text matching instead of semantic search
    matching_memories = [
        memory
        for memory in all_memories
        if search_data.query.lower() in memory.content.lower()
    ]

    # Return up to k results
    results = matching_memories[: search_data.k]

    return [
        MemoryResponse(
            content=memory.content,
            metadata=memory.metadata,
            timestamp=memory.timestamp.isoformat(),
        )
        for memory in results
    ]


@app.delete("/memories")
async def clear_memories():
    """Clear all memories"""
    global memory_engine
    memory_engine = SimpleMemoryEngine()
    return {"message": "All memories cleared", "count": 0}


@app.get("/stats")
async def get_stats():
    """Get memory statistics"""
    return {
        "total_memories": memory_engine.get_memory_count(),
        "status": "operational",
        "features": {
            "basic_storage": True,
            "text_search": True,
            "vector_search": False,  # Disabled due to numpy issues
            "persistence": False,  # Not implemented in simple version
        },
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting AI Memory Layer Simple API...")
    print("API documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
