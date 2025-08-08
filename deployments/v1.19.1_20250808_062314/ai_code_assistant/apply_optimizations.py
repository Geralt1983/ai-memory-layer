#!/usr/bin/env python3
"""
Apply Optimizations to AI Code Assistant
========================================

This script updates the existing implementation with vector-driven search.
"""

import os
import sys
from pathlib import Path

def apply_optimizations():
    """Apply the optimizations to main.py"""
    
    # Read current main.py
    main_py = Path("main.py")
    if not main_py.exists():
        print("Error: main.py not found. Run this from the ai_code_assistant directory.")
        return
    
    with open(main_py, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = Path("main.py.backup")
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"Created backup: {backup_path}")
    
    # Add imports at the top
    new_imports = """
# Optimized imports
from optimized_memory_loader import process_all_commits, OptimizedCommitProcessor
from optimized_query_handler import OptimizedQueryHandler
"""
    
    # Find the imports section and add our imports
    import_marker = "from gpt_assistant import"
    if import_marker in content:
        content = content.replace(
            import_marker,
            new_imports + "\n" + import_marker
        )
    
    # Replace the startup event to use optimized loader
    old_startup = """async def startup():
    \"\"\"Initialize services on startup\"\"\"
    logger.info("🚀 Starting AI Code Assistant")
    logger.info(f"   Sync directory: {SYNC_DIR}")
    logger.info(f"   Data directory: {DATA_DIR}")
    logger.info(f"   OpenAI configured: {bool(OPENAI_API_KEY)}")
    
    # Process any existing commits
    await process_existing_commits()"""
    
    new_startup = """async def startup():
    \"\"\"Initialize services on startup\"\"\"
    logger.info("🚀 Starting AI Code Assistant")
    logger.info(f"   Sync directory: {SYNC_DIR}")
    logger.info(f"   Data directory: {DATA_DIR}")
    logger.info(f"   OpenAI configured: {bool(OPENAI_API_KEY)}")
    
    # Process commits with optimized chunking
    total_chunks = await process_all_commits(SYNC_DIR, vector_store, embedding_service)
    logger.info(f"📚 Loaded {total_chunks} semantic chunks from commits")"""
    
    if old_startup in content:
        content = content.replace(old_startup, new_startup)
    
    # Replace the query endpoint with optimized handler
    old_query_start = """@app.post("/query", response_model=QueryResponse)
async def query_assistant(request: QueryRequest) -> QueryResponse:"""
    
    # Find the query endpoint and note its position
    if old_query_start in content:
        # Add initialization of query handler before the endpoint
        handler_init = """
# Initialize optimized query handler
query_handler = OptimizedQueryHandler(
    memory_query=memory_query,
    prompt_builder=prompt_builder,
    gpt_assistant=gpt_assistant,
    max_context_size=5,  # Only use top 5 most relevant chunks
    enable_reranking=True
)

@app.post("/query", response_model=QueryResponse)
async def query_assistant(request: QueryRequest) -> QueryResponse:
    \"\"\"Process a query with optimized vector search\"\"\"
    try:
        result = await query_handler.handle_query(request.query)
        
        return QueryResponse(
            response=result["response"],
            sources=result["sources"],
            processing_time=result["metrics"]["retrieval_time"]
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))"""
        
        # Find the end of the current query function
        query_end = content.find("\n\n@app.", content.find(old_query_start))
        if query_end == -1:
            query_end = content.find("\n\nif __name__", content.find(old_query_start))
        
        if query_end != -1:
            # Replace the entire query function
            query_start = content.find("@app.post(\"/query\"")
            content = content[:query_start] + handler_init + content[query_end:]
    
    # Save optimized version
    with open("main_optimized.py", 'w') as f:
        f.write(content)
    
    print("\n✅ Optimizations applied!")
    print("\nNext steps:")
    print("1. Review main_optimized.py")
    print("2. Test the changes")
    print("3. If everything works, rename:")
    print("   mv main.py main_old.py")
    print("   mv main_optimized.py main.py")
    print("\n⚡ Expected improvements:")
    print("- Response time: 24s → 3-7s")
    print("- Memory usage: Load only top-5 relevant chunks")
    print("- Answer quality: More focused and specific")

if __name__ == "__main__":
    apply_optimizations()