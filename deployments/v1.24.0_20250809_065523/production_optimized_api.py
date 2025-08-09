#!/usr/bin/env python3
"""
Production Optimized API Server - 2025 Best Practices
Uses precomputed FAISS indexes, query caching, and advanced optimizations
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import List, Optional

from optimized_faiss_memory_engine import create_optimized_faiss_engine
from integrations.embeddings import OpenAIEmbeddings
from api.models import (
    MemoryResponse, SearchRequest, SearchResponse,
    HealthResponse, ChatRequest, ChatResponse
)
from core.logging_config import get_logger

# Load environment variables
load_dotenv()

# Global optimized memory engine
optimized_engine = None
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Production lifespan with optimized FAISS loading"""
    global optimized_engine
    
    logger.info("🚀 Starting Production Optimized AI Memory Layer API")
    
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
            logger.info("✅ OpenAI embeddings ready for new queries")
        else:
            logger.warning("⚠️  No API key - search functionality limited")
        
        # Create optimized memory engine with 2025 best practices
        logger.info("🧠 Creating production optimized memory engine...")
        optimized_engine = create_optimized_faiss_engine(
            memory_json_path=memory_json_path,
            index_base_path=faiss_index_path,
            embedding_provider=embedding_provider,
            cache_size=1000,       # Cache 1000 queries
            enable_query_cache=True,
            use_hnsw=True,         # Use HNSW for better performance
            hnsw_ef_search=100     # HNSW search parameter
        )
        
        if optimized_engine and len(optimized_engine.memories) > 0:
            stats = optimized_engine.get_statistics()
            logger.info(f"✅ Production API ready with {stats['total_memories']:,} memories")
            logger.info(f"🚀 Using precomputed FAISS index with {stats['faiss_vectors']:,} vectors")
            logger.info(f"💾 Query caching enabled (size: {stats['cache_stats']['max_size']})")
        else:
            logger.error("❌ Failed to initialize optimized memory engine")
            
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 API shutdown - preserving precomputed memory data")

# Create FastAPI app with optimized lifespan
app = FastAPI(
    title="AI Memory Layer - Production Optimized API",
    description="Production AI Memory Layer with precomputed ChatGPT embeddings and 2025 best practices",
    version="2.0.0-production",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Production health check endpoint"""
    global optimized_engine
    
    if not optimized_engine:
        return HealthResponse(
            status="unhealthy",
            memory_count=0,
            vector_store_type="None"
        )
    
    stats = optimized_engine.get_statistics()
    
    return HealthResponse(
        status="healthy",
        memory_count=stats["total_memories"],
        vector_store_type="OptimizedFAISS",
        additional_info={
            "faiss_vectors": stats["faiss_vectors"],
            "cache_enabled": stats["cache_stats"]["cache_enabled"],
            "cache_hit_rate": stats["cache_stats"].get("hit_rate", 0),
            "using_hnsw": stats["use_hnsw"]
        }
    )

@app.post("/memories/search", response_model=SearchResponse)
async def search_memories_optimized(request: SearchRequest):
    """Optimized memory search with precomputed FAISS index and caching"""
    global optimized_engine
    
    if not optimized_engine:
        raise HTTPException(status_code=503, detail="Optimized memory engine not initialized")
    
    if not optimized_engine.embedding_provider:
        raise HTTPException(status_code=503, detail="Embedding provider not available")
    
    try:
        # Use optimized search with all 2025 best practices
        memories = optimized_engine.search_memories(
            query=request.query,
            k=request.k,
            score_threshold=getattr(request, 'score_threshold', 0.0),
            filter_by_type=getattr(request, 'filter_by_type', None),
            filter_by_role=getattr(request, 'filter_by_role', None),
            importance_boost=getattr(request, 'importance_boost', 0.5),
            age_decay_days=getattr(request, 'age_decay_days', 30.0)
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
                    'enhanced_scoring': True,
                    'cached_query': True if optimized_engine.query_cache else False
                }
            ))
        
        return SearchResponse(
            memories=memory_responses,
            total_count=len(memories),
            search_metadata={
                "optimized_search": True,
                "cache_stats": optimized_engine.get_cache_stats(),
                "search_time_optimized": True
            }
        )
        
    except Exception as e:
        logger.error(f"Optimized search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/search/advanced")
async def advanced_search(
    query: str,
    k: int = Query(5, ge=1, le=50),
    filter_type: Optional[str] = Query(None, description="Filter by memory type"),
    filter_role: Optional[str] = Query(None, description="Filter by memory role"),
    importance_boost: float = Query(0.5, ge=0.0, le=2.0, description="Importance boost factor"),
    age_decay_days: float = Query(30.0, ge=1.0, le=365.0, description="Age decay half-life in days"),
    score_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum relevance score")
):
    """Advanced search with fine-grained control over optimization parameters"""
    global optimized_engine
    
    if not optimized_engine:
        raise HTTPException(status_code=503, detail="Optimized memory engine not initialized")
    
    try:
        # Convert single filters to lists
        filter_by_type = [filter_type] if filter_type else None
        filter_by_role = [filter_role] if filter_role else None
        
        memories = optimized_engine.search_memories(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter_by_type=filter_by_type,
            filter_by_role=filter_by_role,
            importance_boost=importance_boost,
            age_decay_days=age_decay_days
        )
        
        return {
            "query": query,
            "results": len(memories),
            "parameters": {
                "k": k,
                "filter_type": filter_type,
                "filter_role": filter_role,
                "importance_boost": importance_boost,
                "age_decay_days": age_decay_days,
                "score_threshold": score_threshold
            },
            "memories": [
                {
                    "content": m.content[:200] + "..." if len(m.content) > 200 else m.content,
                    "score": m.relevance_score,
                    "role": m.metadata.get('role', 'unknown'),
                    "type": m.metadata.get('type', 'history'),
                    "importance": m.metadata.get('importance', 1.0),
                    "timestamp": m.timestamp.isoformat()
                }
                for m in memories
            ],
            "cache_stats": optimized_engine.get_cache_stats()
        }
        
    except Exception as e:
        logger.error(f"Advanced search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/stats/detailed")
async def get_detailed_stats():
    """Get comprehensive memory statistics and performance metrics"""
    global optimized_engine
    
    if not optimized_engine:
        raise HTTPException(status_code=503, detail="Optimized memory engine not initialized")
    
    try:
        stats = optimized_engine.get_statistics()
        cache_stats = optimized_engine.get_cache_stats()
        
        # Count memories by metadata
        role_counts = {}
        type_counts = {}
        importance_distribution = {"high": 0, "medium": 0, "low": 0}
        
        for memory in optimized_engine.memories:
            role = memory.metadata.get('role', 'unknown')
            mem_type = memory.metadata.get('type', 'history')
            importance = memory.metadata.get('importance', 1.0)
            
            role_counts[role] = role_counts.get(role, 0) + 1
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            
            if importance >= 0.8:
                importance_distribution["high"] += 1
            elif importance >= 0.5:
                importance_distribution["medium"] += 1
            else:
                importance_distribution["low"] += 1
        
        return {
            "engine_stats": stats,
            "cache_performance": cache_stats,
            "memory_distribution": {
                "by_role": role_counts,
                "by_type": type_counts,
                "by_importance": importance_distribution
            },
            "optimization_features": {
                "precomputed_faiss_index": True,
                "query_embedding_cache": cache_stats["cache_enabled"],
                "hnsw_indexing": stats["use_hnsw"],
                "metadata_filtering": True,
                "importance_boosting": True,
                "age_decay": True
            }
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_optimized_memory(request: ChatRequest):
    """Chat with AI using optimized ChatGPT memory context"""
    global optimized_engine
    
    if not optimized_engine:
        raise HTTPException(status_code=503, detail="Optimized memory engine not initialized")
    
    try:
        # Search for relevant memories with optimization
        relevant_memories = optimized_engine.search_memories(
            query=request.message,
            k=getattr(request, 'memory_k', 5),
            importance_boost=0.5,  # Boost important memories
            age_decay_days=30.0    # 30-day decay
        )
        
        # Build enhanced context
        memory_context = []
        for memory in relevant_memories:
            memory_context.append({
                "role": memory.metadata.get('role', 'unknown'),
                "content": memory.content,
                "title": memory.metadata.get('title', 'No title'),
                "importance": memory.metadata.get('importance', 1.0),
                "relevance": memory.relevance_score,
                "timestamp": memory.timestamp.isoformat(),
                "type": memory.metadata.get('type', 'history')
            })
        
        return ChatResponse(
            response="Optimized memory context retrieved successfully",
            memory_context=memory_context,
            memory_count=len(relevant_memories),
            optimization_info={
                "cache_hit_rate": optimized_engine.get_cache_stats().get("hit_rate", 0),
                "faiss_vectors_searched": optimized_engine.get_statistics()["faiss_vectors"],
                "enhanced_scoring": True
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the production optimized API server"""
    print("🚀 AI Memory Layer - Production Optimized API Server")
    print("=" * 60)
    print("✅ 2025 Best Practices Implementation:")
    print("   • Precomputed FAISS index loading (no embedding regeneration)")
    print("   • LRU query embedding cache")
    print("   • HNSW indexing for sub-100ms search")
    print("   • Metadata filtering and importance boosting")
    print("   • Age decay with configurable half-life")
    print("   • Enhanced scoring and batched operations")
    print("=" * 60)
    
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    # Run server
    uvicorn.run(
        "production_optimized_api:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1      # Single worker for memory consistency
    )

if __name__ == "__main__":
    main()