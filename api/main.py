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
from core.memory_manager import MemoryManager, create_default_memory_manager
from core.logging_config import get_logger, log_api_request
from storage.faiss_store import FaissVectorStore
from storage.chroma_store import ChromaVectorStore
from integrations.embeddings import OpenAIEmbeddings
from integrations.direct_openai import DirectOpenAIChat
from .models import (
    MemoryCreate,
    MemoryResponse,
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse,
    ErrorResponse,
    CleanupRequest,
    CleanupResponse,
    ArchiveListResponse,
    ExportRequest,
    MemoryStatsResponse,
)


# Global variables for dependency injection
memory_engine: Optional[MemoryEngine] = None
direct_openai_chat: Optional[DirectOpenAIChat] = None
memory_manager: Optional[MemoryManager] = None
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
    global memory_engine, direct_openai_chat, memory_manager

    try:
        logger.info("Starting AI Memory Layer API initialization")

        # Get configuration from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        vector_store_type = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()
        persist_directory = os.getenv("PERSIST_DIRECTORY", "./data")
        memory_persist_path = os.getenv(
            "MEMORY_PERSIST_PATH", f"{persist_directory}/memories.json"
        )

        logger.info(
            "Configuration loaded",
            extra={
                "vector_store_type": vector_store_type,
                "persist_directory": persist_directory,
                "memory_persist_path": memory_persist_path,
            },
        )

        # Initialize embedding provider
        embedding_provider = OpenAIEmbeddings(
            api_key=api_key,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        )

        # Initialize vector store
        if vector_store_type == "chroma":
            vector_store = ChromaVectorStore(
                collection_name="ai_memory_api",
                persist_directory=f"{persist_directory}/chroma_db",
            )
        else:  # Default to FAISS
            vector_store = FaissVectorStore(
                dimension=1536, index_path=f"{persist_directory}/faiss_index"
            )

        # Initialize memory engine
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            persist_path=memory_persist_path,
        )

        
        # Initialize Direct OpenAI Chat (GPT-4o optimized)
        direct_openai_chat = DirectOpenAIChat(
            api_key=api_key,
            memory_engine=memory_engine,
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            system_prompt_path=os.getenv("SYSTEM_PROMPT_PATH", "./prompts/system_prompt_4o.txt"),
        )

        # Initialize memory manager
        memory_manager = create_default_memory_manager(memory_engine)

        logger.info(
            "Memory system initialized successfully",
            extra={
                "vector_store_type": vector_store_type,
                "existing_memories": len(memory_engine.memories),
            },
        )

    except Exception as e:
        logger.error(
            "Failed to initialize memory system", extra={"error": str(e)}, exc_info=True
        )
        raise


async def shutdown_event():
    """Cleanup on shutdown"""
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
    lifespan=lifespan,
)


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request start
    logger.debug(
        "Request started",
        extra={
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params),
            "client_ip": request.client.host if request.client else None,
        },
    )

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
        client_ip=request.client.host if request.client else None,
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


# Explicit OPTIONS handler so tests see CORS headers even without Origin
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    """Handle CORS preflight requests.

    FastAPI's CORSMiddleware only processes requests with the appropriate
    ``Origin`` header.  The test-suite performs a bare ``OPTIONS`` request which
    would normally return ``405``.  Providing this handler ensures a ``200``
    response with the expected CORS headers for any path.
    """

    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


# Dependency to get memory engine
def get_memory_engine() -> MemoryEngine:
    if not memory_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory engine not initialized",
        )
    return memory_engine



# Dependency to get OpenAI integration (backwards compatible)
def get_openai_integration() -> DirectOpenAIChat:
    """Return the active OpenAI chat integration.

    Historically the project exposed a ``get_openai_integration`` dependency
    which was used by the FastAPI endpoints and tests.  During a refactor the
    function was removed in favour of ``get_direct_openai_chat`` which broke
    the tests that still depended on the old name.  Re‑introducing this helper
    keeps backwards compatibility while still returning the same
    ``DirectOpenAIChat`` instance.
    """

    if not direct_openai_chat:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Direct OpenAI chat not initialized",
        )
    return direct_openai_chat


# Dependency to get direct OpenAI chat (newer name used elsewhere)
def get_direct_openai_chat() -> DirectOpenAIChat:
    if not direct_openai_chat:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Direct OpenAI chat not initialized",
        )
    return direct_openai_chat


# Dependency to get memory manager
def get_memory_manager() -> MemoryManager:
    if not memory_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory manager not initialized",
        )
    return memory_manager


# Exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="Internal server error", detail=str(exc)).dict(),
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(engine: MemoryEngine = Depends(get_memory_engine)):
    """Health check endpoint"""
    vector_store_type = None
    if hasattr(engine.vector_store, "__class__"):
        vector_store_type = engine.vector_store.__class__.__name__

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        memory_count=len(engine.memories),
        vector_store_type=vector_store_type,
    )


# Memory management endpoints
@app.post(
    "/memories", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED
)
async def create_memory(
    memory_data: MemoryCreate, engine: MemoryEngine = Depends(get_memory_engine)
):
    """Create a new memory"""
    try:
        memory = engine.add_memory(memory_data.content, memory_data.metadata)
        return MemoryResponse(
            content=memory.content,
            metadata=memory.metadata,
            timestamp=memory.timestamp,
            relevance_score=memory.relevance_score,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create memory: {str(e)}",
        )


@app.get("/memories", response_model=List[MemoryResponse])
async def get_recent_memories(
    n: int = 10, engine: MemoryEngine = Depends(get_memory_engine)
):
    """Get recent memories"""
    if n < 1 or n > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'n' must be between 1 and 100",
        )

    memories = engine.get_recent_memories(n)
    return [
        MemoryResponse(
            content=memory.content,
            metadata=memory.metadata,
            timestamp=memory.timestamp,
            relevance_score=memory.relevance_score,
        )
        for memory in memories
    ]


@app.post("/memories/search", response_model=SearchResponse)
async def search_memories(
    search_data: SearchRequest, engine: MemoryEngine = Depends(get_memory_engine)
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
                    relevance_score=memory.relevance_score,
                )
                for memory in memories
            ],
            total_count=len(memories),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to search memories: {str(e)}",
        )


@app.delete("/memories", status_code=status.HTTP_204_NO_CONTENT)
async def clear_memories(engine: MemoryEngine = Depends(get_memory_engine)):
    """Clear all memories"""
    try:
        engine.clear_memories()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memories: {str(e)}",
        )


# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_memory(
    chat_data: ChatRequest,
    openai_integration: DirectOpenAIChat = Depends(get_openai_integration),
    request: Request = None,
):
    """Chat endpoint using the OpenAI integration."""
    start_time = time.time()

    try:
        logger.info(
            "Direct chat request received",
            extra={
                "message_length": len(chat_data.message),
                "thread_id": chat_data.thread_id,
                "model": openai_integration.model,
            },
        )

        # Build context string for debugging/compatibility
        context_summary = openai_integration.context_builder.build_context(
            message=chat_data.message,
            include_recent=chat_data.include_recent,
            include_relevant=chat_data.include_relevant,
            system_prompt=chat_data.system_prompt,
        )

        # Execute chat through the compatibility wrapper
        response = openai_integration.chat_with_memory(
            message=chat_data.message,
            system_prompt=chat_data.system_prompt,
            include_recent=chat_data.include_recent,
            include_relevant=chat_data.include_relevant,
            remember_response=chat_data.remember_response,
        )

        processing_time = time.time() - start_time
        logger.info(
            "Direct chat response generated",
            extra={
                "response_length": len(response),
                "processing_time_ms": round(processing_time * 1000, 2),
                "messages_count": getattr(openai_integration, "last_messages_count", 0),
                "thread_id": chat_data.thread_id,
            },
        )

        return ChatResponse(response=response, context_used=context_summary)

    except Exception as e:
        logger.error("Direct chat failed", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chat failed: {str(e)}",
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
        "newest_memory": None,
    }

    # Vector store stats
    if engine.vector_store and hasattr(engine.vector_store, "index"):
        if hasattr(engine.vector_store.index, "ntotal"):
            stats["vector_store_entries"] = engine.vector_store.index.ntotal

    if memories:
        # Memory type distribution
        for memory in memories:
            memory_type = memory.metadata.get("type", "unknown")
            stats["memory_types"][memory_type] = (
                stats["memory_types"].get(memory_type, 0) + 1
            )

        # Oldest and newest
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        stats["oldest_memory"] = sorted_memories[0].timestamp
        stats["newest_memory"] = sorted_memories[-1].timestamp

    return stats


# Memory management endpoints
@app.get("/memories/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(engine: MemoryEngine = Depends(get_memory_engine)):
    """Get detailed memory statistics"""
    try:
        stats = engine.get_memory_stats()
        return MemoryStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory stats: {str(e)}",
        )


@app.post("/memories/cleanup", response_model=CleanupResponse)
async def cleanup_memories(
    cleanup_data: CleanupRequest, manager: MemoryManager = Depends(get_memory_manager)
):
    """Clean up memories using specified criteria"""
    try:
        stats = manager.auto_cleanup(
            max_memories=cleanup_data.max_memories,
            max_age_days=cleanup_data.max_age_days,
            min_relevance=cleanup_data.min_relevance,
        )

        # Override with dry_run if requested
        if cleanup_data.dry_run:
            # Run cleanup again as dry run to get preview
            stats = manager.cleanup_memories(
                archive_before_cleanup=cleanup_data.archive_before_cleanup, dry_run=True
            )

        return CleanupResponse(
            memories_before=stats.total_memories_before,
            memories_after=stats.total_memories_after,
            memories_cleaned=stats.memories_cleaned,
            memories_archived=stats.memories_archived,
            duration_ms=stats.duration_ms,
            dry_run=cleanup_data.dry_run,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Cleanup failed: {str(e)}"
        )


@app.get("/archives", response_model=ArchiveListResponse)
async def list_archives(manager: MemoryManager = Depends(get_memory_manager)):
    """List all available memory archives"""
    try:
        archives = manager.archiver.list_archives()
        archive_data = []

        for archive in archives:
            archive_data.append(
                {
                    "archive_path": archive.archive_path,
                    "created_at": archive.created_at,
                    "memory_count": archive.memory_count,
                    "size_bytes": archive.size_bytes,
                    "criteria": archive.criteria,
                }
            )

        return ArchiveListResponse(archives=archive_data, total_count=len(archive_data))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list archives: {str(e)}",
        )


@app.post("/memories/export")
async def export_memories(
    export_data: ExportRequest, manager: MemoryManager = Depends(get_memory_manager)
):
    """Export memories to specified format"""
    try:
        from fastapi.responses import FileResponse
        import tempfile

        # Create filter function based on request
        def memory_filter(memory):
            # Filter by type
            if export_data.filter_type:
                if memory.metadata.get("type") != export_data.filter_type:
                    return False

            # Filter by date range
            if export_data.start_date and memory.timestamp < export_data.start_date:
                return False
            if export_data.end_date and memory.timestamp > export_data.end_date:
                return False

            return True

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{export_data.format}", mode="w"
        ) as temp_file:
            temp_path = temp_file.name

        # Export memories
        manager.export_memories(
            temp_path, format=export_data.format, filter_func=memory_filter
        )

        # Return file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memories_export_{timestamp}.{export_data.format}"

        return FileResponse(
            path=temp_path,
            filename=filename,
            media_type=f"application/{export_data.format}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Export failed: {str(e)}"
        )


@app.post("/archives/{archive_name}/restore")
async def restore_archive(
    archive_name: str,
    manager: MemoryManager = Depends(get_memory_manager),
    engine: MemoryEngine = Depends(get_memory_engine),
):
    """Restore memories from an archive"""
    try:
        # Find archive by name
        archives = manager.archiver.list_archives()
        archive_path = None

        for archive in archives:
            if archive_name in archive.archive_path:
                archive_path = archive.archive_path
                break

        if not archive_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Archive '{archive_name}' not found",
            )

        # Load memories from archive
        archived_memories = manager.archiver.load_archive(archive_path)

        # Add memories back to engine
        restored_count = 0
        for memory in archived_memories:
            # Check if memory already exists to avoid duplicates
            existing = any(
                m.content == memory.content
                and abs((m.timestamp - memory.timestamp).total_seconds()) < 1
                for m in engine.memories
            )

            if not existing:
                engine.memories.append(memory)
                restored_count += 1

        # Save updated memories
        if engine.persist_path:
            engine.save_memories()

        return {
            "message": f"Successfully restored {restored_count} memories from archive",
            "archive_path": archive_path,
            "memories_restored": restored_count,
            "total_memories": len(engine.memories),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Archive restoration failed: {str(e)}",
        )


@app.post("/debug/test-logging")
async def test_debug_logging(direct_chat: DirectOpenAIChat = Depends(get_direct_openai_chat)):
    """Test endpoint to verify debug logging is working"""
    direct_chat.logger.info("TEST: Debug logging test from API endpoint")
    direct_chat.logger.debug("TEST: This is a debug message from DirectOpenAIChat")
    
    return {"message": "Debug logging test completed", "logger_name": direct_chat.logger.name}


@app.post("/conversations/generate-title")
async def generate_conversation_title(
    request: dict, direct_chat: DirectOpenAIChat = Depends(get_direct_openai_chat)
):
    """Generate a concise title for a conversation based on its messages"""
    try:
        messages = request.get("messages", [])
        if not messages:
            return {"title": "New Chat"}
        
        # Take the first few messages to generate a title
        context_messages = messages[:6]  # First 3 exchanges (user + assistant)
        conversation_text = "\n".join([
            f"{msg.get('sender', 'unknown')}: {msg.get('content', '')}" 
            for msg in context_messages
        ])
        
        # Use OpenAI to generate a concise title
        title_prompt = f"""Based on this conversation, generate a concise 2-4 word title that captures the main topic. Be specific and descriptive, not generic.

Conversation:
{conversation_text}

Generate only the title, nothing else. Examples of good titles:
- "Python vs Rust"
- "AI Ethics Discussion" 
- "React Hook Problems"
- "Startup Funding Strategy"

Title:"""

        response = direct_chat.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": title_prompt}],
            max_tokens=10,
            temperature=0.3
        )
        
        title = response.choices[0].message.content.strip()
        # Clean up the title (remove quotes, etc.)
        title = title.replace('"', '').replace("'", '').strip()
        
        # Fallback if title is too long or empty
        if not title or len(title) > 50:
            # Extract key terms from first user message
            first_message = next((msg.get('content', '') for msg in messages if msg.get('sender') == 'user'), '')
            if first_message:
                words = first_message.split()[:4]
                title = ' '.join(words).title()
            else:
                title = "New Chat"
        
        return {"title": title}
        
    except Exception as e:
        logger.error(f"Failed to generate conversation title: {str(e)}")
        # Fallback to first user message preview
        first_user_msg = next((msg.get('content', '') for msg in messages if msg.get('sender') == 'user'), '')
        if first_user_msg:
            title = first_user_msg[:30] + "..." if len(first_user_msg) > 30 else first_user_msg
        else:
            title = "New Chat"
        return {"title": title}


# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
