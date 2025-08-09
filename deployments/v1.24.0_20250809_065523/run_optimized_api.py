#!/usr/bin/env python3
"""
Optimized API Server
Uses pre-computed FAISS embeddings for instant startup
Based on 2025 best practices for vector database optimization
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from optimized_memory_engine import create_optimized_memory_engine
from integrations.embeddings import OpenAIEmbeddings
from api.models import (
    MemoryRequest, MemoryResponse, SearchRequest, SearchResponse,
    HealthResponse, ChatRequest, ChatResponse
)
from core.logging_config import get_logger

# Load environment variables
load_dotenv()

# Global memory engine
memory_engine = None
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized application lifespan with fast startup"""
    global memory_engine
    
    logger.info("🚀 Starting Optimized AI Memory Layer API")
    
    try:
        # Configuration
        memory_json_path = os.getenv("MEMORY_PERSIST_PATH", "./data/chatgpt_memories.json")
        faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_chatgpt_index")
        api_key = os.getenv("OPENAI_API_KEY")
        
        logger.info(f"📁 Memory JSON: {memory_json_path}")
        logger.info(f"🔍 FAISS Index: {faiss_index_path}")
        
        # Initialize embedding provider (for new queries only)
        embedding_provider = None
        if api_key:
            embedding_provider = OpenAIEmbeddings(api_key)
            logger.info("✅ OpenAI embeddings ready for queries")
        else:
            logger.warning("⚠️  No API key - search functionality limited")
        
        # Create optimized memory engine with pre-computed embeddings
        logger.info("🧠 Creating optimized memory engine...")
        memory_engine = create_optimized_memory_engine(
            memory_json_path=memory_json_path,
            faiss_index_path=faiss_index_path,
            embedding_provider=embedding_provider,
            auto_save=False  # Prevent overwriting ChatGPT data
        )
        
        if memory_engine and len(memory_engine.memories) > 0:
            stats = memory_engine.get_statistics()
            logger.info(f"✅ Optimized API ready with {stats['total_memories']} memories")
            logger.info("🎯 Using pre-computed FAISS embeddings for instant search")
        else:
            logger.error("❌ Failed to initialize optimized memory engine")
            
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    
    yield
    
    # Cleanup (don't save to prevent overwriting precomputed data)
    logger.info("🔄 API shutdown - preserving precomputed memory data")

# Create FastAPI app with optimized lifespan
app = FastAPI(
    title="AI Memory Layer - Optimized API",
    description="Optimized AI Memory Layer with pre-computed ChatGPT embeddings",
    version="1.12.0-optimized",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global memory_engine
    
    memory_count = len(memory_engine.memories) if memory_engine else 0
    vector_store_type = type(memory_engine.vector_store).__name__ if memory_engine else "None"
    
    return HealthResponse(
        status="healthy" if memory_engine else "unhealthy",
        memory_count=memory_count,
        vector_store_type=vector_store_type
    )

@app.post("/memories", response_model=MemoryResponse)
async def create_memory(request: MemoryRequest):
    """Create a new memory (not recommended with precomputed data)"""
    global memory_engine
    
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Memory engine not initialized")
    
    logger.warning("Adding new memory to precomputed dataset - consider rebuilding FAISS index")
    
    try:
        memory = memory_engine.add_memory(
            content=request.content,
            role=getattr(request, 'role', 'user'),
            thread_id=getattr(request, 'thread_id', None),
            title=getattr(request, 'title', None),
            type=getattr(request, 'type', 'history'),
            importance=getattr(request, 'importance', 1.0),
            metadata=request.metadata or {}
        )
        
        return MemoryResponse(
            id=str(len(memory_engine.memories) - 1),
            content=memory.content,
            timestamp=memory.timestamp,
            relevance_score=memory.relevance_score,
            metadata=memory.metadata
        )
        
    except Exception as e:
        logger.error(f"Error creating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/search", response_model=SearchResponse)
async def search_memories(request: SearchRequest):
    """Search memories using optimized pre-computed embeddings"""
    global memory_engine
    
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Memory engine not initialized")
    
    if not memory_engine.embedding_provider:
        raise HTTPException(status_code=503, detail="Embedding provider not available")
    
    try:
        memories = memory_engine.search_memories(
            query=request.query,
            k=request.k,
            score_threshold=getattr(request, 'score_threshold', 0.0)
        )
        
        memory_responses = []
        for i, memory in enumerate(memories):
            memory_responses.append(MemoryResponse(
                id=str(i),
                content=memory.content,
                timestamp=memory.timestamp,
                relevance_score=memory.relevance_score,
                metadata={
                    **memory.metadata,
                    'role': memory.role,
                    'thread_id': memory.thread_id,
                    'title': memory.title,
                    'type': memory.type,
                    'importance': memory.importance
                }
            ))
        
        return SearchResponse(
            memories=memory_responses,
            total_count=len(memories)
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/stats")
async def get_memory_stats():
    """Get detailed memory statistics"""
    global memory_engine
    
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Memory engine not initialized")
    
    try:
        stats = memory_engine.get_statistics()
        return {
            "total_memories": stats["total_memories"],
            "faiss_vector_count": stats.get("faiss_vector_count", 0),
            "precomputed_mode": stats.get("precomputed_mode", False),
            "auto_save_enabled": stats.get("auto_save_enabled", False),
            "oldest_memory": stats["oldest_memory"],
            "newest_memory": stats["newest_memory"],
            "average_content_length": stats["average_content_length"]
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_memory(request: ChatRequest):
    """Chat with AI using ChatGPT memory context"""
    global memory_engine
    
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Memory engine not initialized")
    
    try:
        # Search for relevant memories
        relevant_memories = memory_engine.search_memories(
            query=request.message,
            k=getattr(request, 'memory_k', 5)
        )
        
        # Build context from memories
        memory_context = []
        for memory in relevant_memories:
            memory_context.append({
                "role": memory.role,
                "content": memory.content,
                "title": memory.title,
                "importance": memory.importance,
                "relevance": memory.relevance_score
            })
        
        # For now, return the memory context (can be extended with actual chat)
        return ChatResponse(
            response="Memory context retrieved successfully",
            memory_context=memory_context,
            memory_count=len(relevant_memories)
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the optimized API server"""
    print("🚀 AI Memory Layer - Optimized API Server")
    print("=" * 50)
    print("✅ Using pre-computed FAISS embeddings")
    print("⚡ Instant startup with 23K+ ChatGPT memories")
    print("🔍 Optimized search with importance weighting")
    print("=" * 50)
    
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    # Run server
    uvicorn.run(
        "run_optimized_api:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1  # Single worker for memory consistency
    )

if __name__ == "__main__":
    main()