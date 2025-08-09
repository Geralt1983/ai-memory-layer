#!/usr/bin/env python3
"""
AI Memory Layer - Modular API Server (v1.3.0)
Refactored for clean architecture with separated endpoints
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv('.env.local')

# Import optimized loader and GPT response generation
from optimized_memory_loader import create_optimized_chatgpt_engine
from optimized_clean_loader import create_cleaned_chatgpt_engine
from core.similarity_utils import create_search_optimized_engine

# Try to load cleaned memories first, fall back to original if not available
print("🚀 Loading ChatGPT Memory System...")
try:
    if Path("data/chatgpt_memories_cleaned.json").exists():
        print("🧹 Using CLEANED memories for better quality...")
        memory_engine = create_cleaned_chatgpt_engine()
    else:
        print("📂 Using original memories...")
        memory_engine = create_optimized_chatgpt_engine()
except Exception as e:
    print(f"⚠️ Failed to load cleaned memories: {e}")
    print("📂 Falling back to original memories...")
    memory_engine = create_optimized_chatgpt_engine()

# Enhance memory engine with relevance scoring
print("🎯 Adding semantic relevance scoring...")
memory_engine = create_search_optimized_engine(memory_engine, min_score=0.4)
    
print(f"✅ {len(memory_engine.memories):,} memories loaded with enhanced search!")

# Create FastAPI app
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="AI Memory Layer API", 
    version="1.3.0",
    description="Modular API for ChatGPT conversation memory with semantic search"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to inject memory engine
def get_memory_engine():
    return memory_engine

# Root route - serve the enhanced chat interface with metrics
@app.get("/", response_class=HTMLResponse)
async def root():
    # Read and return the enhanced web interface
    static_path = project_root / "static" / "web_interface.html"
    with open(static_path, 'r') as f:
        html_content = f.read()
    # Update the API base URL to work with tunnel
    html_content = html_content.replace("const API_BASE = 'http://localhost:8000';", "const API_BASE = window.location.origin;")
    return html_content

# Include endpoint routers
from api.endpoints.internal import router as internal_router
app.include_router(internal_router)

@app.get("/health")
async def health(engine = Depends(get_memory_engine)):
    return {
        "status": "healthy",
        "memory_count": len(engine.memories),
        "system": "ai_memory_layer_v1.3.0",
        "dataset_size": f"{len(engine.memories):,} ChatGPT conversations"
    }

if __name__ == "__main__":
    import uvicorn
    print(f"\\n🌐 Starting AI Memory Layer API v1.3.0 with {len(memory_engine.memories):,} memories")
    print("🔗 Available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)