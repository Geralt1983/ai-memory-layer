"""
FastAPI REST API for AI Memory Layer
"""
import os
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time

from core.memory_engine import MemoryEngine
from core.logging_config import get_logger, log_api_request
from storage.faiss_store import FaissVectorStore
from storage.chroma_store import ChromaVectorStore
from integrations.embeddings import OpenAIEmbeddings
from integrations.openai_integration import OpenAIIntegration
from .models import (
    MemoryCreate, MemoryResponse, ChatRequest, ChatResponse,
    SearchRequest, SearchResponse, HealthResponse, ErrorResponse
)


# Global variables for dependency injection
memory_engine: Optional[MemoryEngine] = None
openai_integration: Optional[OpenAIIntegration] = None
logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()


async def startup_event():
    """Initialize the memory system on startup"""
    global memory_engine, openai_integration
    
    try:
        logger.info("Starting AI Memory Layer API initialization")
        
        # Get configuration from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        vector_store_type = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()
        persist_directory = os.getenv("PERSIST_DIRECTORY", "./data")
        memory_persist_path = os.getenv("MEMORY_PERSIST_PATH", f"{persist_directory}/memories.json")
        
        logger.info("Configuration loaded", extra={
            "vector_store_type": vector_store_type,
            "persist_directory": persist_directory,
            "memory_persist_path": memory_persist_path
        })
        
        # Initialize embedding provider
        embedding_provider = OpenAIEmbeddings(
            api_key=api_key,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        )
        
        # Initialize vector store
        if vector_store_type == "chroma":
            vector_store = ChromaVectorStore(
                collection_name="ai_memory_api",
                persist_directory=f"{persist_directory}/chroma_db"
            )
        else:  # Default to FAISS
            vector_store = FaissVectorStore(
                dimension=1536,
                index_path=f"{persist_directory}/faiss_index"
            )
        
        # Initialize memory engine
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            persist_path=memory_persist_path
        )
        
        # Initialize OpenAI integration
        openai_integration = OpenAIIntegration(
            api_key=api_key,
            memory_engine=memory_engine,
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        )
        
        logger.info("Memory system initialized successfully", extra={
            "vector_store_type": vector_store_type,
            "existing_memories": len(memory_engine.memories)
        })
        
    except Exception as e:
        logger.error("Failed to initialize memory system", extra={"error": str(e)}, exc_info=True)
        raise


async def shutdown_event():
    """Cleanup on shutdown"""
    global memory_engine
    logger.info("API shutdown initiated")
    
    if memory_engine and memory_engine.persist_path:
        memory_engine.save_memories()
        logger.info("Memories saved on shutdown")
    
    logger.info("API shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="AI Memory Layer API",
    description="REST API for managing AI conversation memory and context",
    version="0.1.0",
    lifespan=lifespan
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request start
    logger.debug("Request started", extra={
        "method": request.method,
        "path": request.url.path,
        "query": str(request.query_params),
        "client_ip": request.client.host if request.client else None
    })
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Log request completion
    log_api_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        response_time=response_time,
        client_ip=request.client.host if request.client else None
    )
    
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get memory engine
def get_memory_engine() -> MemoryEngine:
    if not memory_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory engine not initialized"
        )
    return memory_engine


# Dependency to get OpenAI integration
def get_openai_integration() -> OpenAIIntegration:
    if not openai_integration:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI integration not initialized"
        )
    return openai_integration


# Exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(engine: MemoryEngine = Depends(get_memory_engine)):
    """Health check endpoint"""
    vector_store_type = None
    if hasattr(engine.vector_store, '__class__'):
        vector_store_type = engine.vector_store.__class__.__name__
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        memory_count=len(engine.memories),
        vector_store_type=vector_store_type
    )


# Memory management endpoints
@app.post("/memories", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory_data: MemoryCreate,
    engine: MemoryEngine = Depends(get_memory_engine)
):
    """Create a new memory"""
    try:
        memory = engine.add_memory(memory_data.content, memory_data.metadata)
        return MemoryResponse(
            content=memory.content,
            metadata=memory.metadata,
            timestamp=memory.timestamp,
            relevance_score=memory.relevance_score
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create memory: {str(e)}"
        )


@app.get("/memories", response_model=List[MemoryResponse])
async def get_recent_memories(
    n: int = 10,
    engine: MemoryEngine = Depends(get_memory_engine)
):
    """Get recent memories"""
    if n < 1 or n > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'n' must be between 1 and 100"
        )
    
    memories = engine.get_recent_memories(n)
    return [
        MemoryResponse(
            content=memory.content,
            metadata=memory.metadata,
            timestamp=memory.timestamp,
            relevance_score=memory.relevance_score
        )
        for memory in memories
    ]


@app.post("/memories/search", response_model=SearchResponse)
async def search_memories(
    search_data: SearchRequest,
    engine: MemoryEngine = Depends(get_memory_engine)
):
    """Search memories by content similarity"""
    try:
        memories = engine.search_memories(search_data.query, k=search_data.k)
        return SearchResponse(
            memories=[
                MemoryResponse(
                    content=memory.content,
                    metadata=memory.metadata,
                    timestamp=memory.timestamp,
                    relevance_score=memory.relevance_score
                )
                for memory in memories
            ],
            total_count=len(memories)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to search memories: {str(e)}"
        )


@app.delete("/memories", status_code=status.HTTP_204_NO_CONTENT)
async def clear_memories(engine: MemoryEngine = Depends(get_memory_engine)):
    """Clear all memories"""
    try:
        engine.clear_memories()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memories: {str(e)}"
        )


# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_memory(
    chat_data: ChatRequest,
    ai: OpenAIIntegration = Depends(get_openai_integration)
):
    """Chat with AI using memory context"""
    try:
        response = ai.chat_with_memory(
            message=chat_data.message,
            system_prompt=chat_data.system_prompt,
            include_recent=chat_data.include_recent,
            include_relevant=chat_data.include_relevant,
            remember_response=chat_data.remember_response
        )
        
        # Get context for response (optional)
        context = None
        try:
            context = ai.context_builder.build_context(
                query=chat_data.message,
                include_recent=chat_data.include_recent,
                include_relevant=chat_data.include_relevant
            )
        except:
            pass  # Context is optional
        
        return ChatResponse(
            response=response,
            context_used=context if context else None
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chat failed: {str(e)}"
        )


# Stats endpoint
@app.get("/stats")
async def get_stats(engine: MemoryEngine = Depends(get_memory_engine)):
    """Get memory statistics"""
    memories = engine.memories
    
    # Basic stats
    stats = {
        "total_memories": len(memories),
        "vector_store_entries": 0,
        "memory_types": {},
        "oldest_memory": None,
        "newest_memory": None
    }
    
    # Vector store stats
    if engine.vector_store and hasattr(engine.vector_store, 'index'):
        if hasattr(engine.vector_store.index, 'ntotal'):
            stats["vector_store_entries"] = engine.vector_store.index.ntotal
    
    if memories:
        # Memory type distribution
        for memory in memories:
            memory_type = memory.metadata.get("type", "unknown")
            stats["memory_types"][memory_type] = stats["memory_types"].get(memory_type, 0) + 1
        
        # Oldest and newest
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        stats["oldest_memory"] = sorted_memories[0].timestamp
        stats["newest_memory"] = sorted_memories[-1].timestamp
    
    return stats


# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )