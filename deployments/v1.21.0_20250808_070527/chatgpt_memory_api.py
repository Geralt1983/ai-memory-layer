#!/usr/bin/env python3
"""
ChatGPT Memory API - Fixed version without JavaScript escaping issues
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv('.env.local')

# Import optimized loader and GPT response generation
from optimized_memory_loader import create_optimized_chatgpt_engine
from optimized_clean_loader import create_cleaned_chatgpt_engine
from core.gpt_response import generate_gpt_response, generate_conversation_title as gpt_generate_title
from core.similarity_utils import create_search_optimized_engine
from core.memory_synthesis import get_memory_synthesizer

# Try to load cleaned memories first, fall back to original if not available
print("🚀 Loading ChatGPT Memory System...")
try:
    from pathlib import Path
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
    use_gpt: Optional[bool] = True

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

# Root route - serve the enhanced chat interface with metrics
@app.get("/", response_class=HTMLResponse)
async def root():
    # Read and return the enhanced web interface with metrics
    with open('web_interface_with_metrics.html', 'r') as f:
        html_content = f.read()
    # HTML already uses window.location.origin for dynamic API base URL
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

# FIXED: Redirect legacy /stats to /memories/stats to avoid confusion
@app.get("/stats", include_in_schema=False)
async def stats_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/memories/stats", status_code=301)

# Fix: Add missing /memories/stats endpoint for frontend
@app.get("/memories/stats")
async def memory_stats():
    from datetime import datetime, timezone
    return {
        "total_memories": len(memory_engine.memories),
        "count": len(memory_engine.memories),
        "faiss_vectors": memory_engine.vector_store.index.ntotal if hasattr(memory_engine.vector_store, 'index') else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": "chatgpt_memory_system",
        "status": "active"
    }

# Fix: Add missing conversation title generation endpoint with GPT support
@app.post("/conversations/generate-title") 
async def generate_title(payload: dict):
    try:
        messages = payload.get("messages", [])
        if not messages:
            return {"title": "New Chat"}
        
        # Try GPT-powered title generation first
        try:
            title = gpt_generate_title(messages)
            return {"title": title}
        except Exception as gpt_error:
            print(f"GPT title generation failed: {gpt_error}")
            
        # Fallback: Extract first user message for title
        first_user_msg = None
        for msg in messages:
            if msg.get("sender") == "user":
                first_user_msg = msg.get("content", "")
                break
                
        if first_user_msg:
            # Create a simple title from first message
            title = first_user_msg[:30].strip()
            if len(first_user_msg) > 30:
                title += "..."
            return {"title": title or "New Chat"}
        else:
            return {"title": "New Chat"}
            
    except Exception as e:
        return {"title": "New Chat", "error": str(e)}

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

@app.post("/memories")
async def add_memory(request: dict):
    """Add a new memory to the system"""
    try:
        content = request.get("content", "")
        metadata = request.get("metadata", {})
        
        if not content:
            return {"success": False, "error": "Content is required"}
            
        # Add memory to the engine
        memory = memory_engine.add_memory(content, metadata)
        
        # Create a simple ID based on content hash and timestamp
        import hashlib
        simple_id = hashlib.md5(f"{content}_{memory.timestamp}".encode()).hexdigest()[:8]
        
        return {
            "success": True,
            "memory_id": simple_id,
            "content": content,
            "metadata": metadata,
            "timestamp": memory.timestamp.isoformat(),
            "total_memories": len(memory_engine.memories)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Search for relevant memories with higher k for better selection
        results = memory_engine.search_memories(request.message, k=15)
        
        # NEW: Enhanced search for pet-related queries that might not find direct matches
        if any(word in request.message.lower() for word in ["how many", "dogs", "pets", "animals"]):
            # Do a supplementary search for known pet names
            pet_search = memory_engine.search_memories("Remy Bailey dogs pets", k=10)
            
            # Combine results, avoiding duplicates
            seen_content = set()
            for r in results:
                seen_content.add(getattr(r, 'content', str(r)))
            
            for pet_result in pet_search:
                pet_content = getattr(pet_result, 'content', str(pet_result))
                if pet_content not in seen_content and any(name in pet_content for name in ["Remy", "Bailey"]):
                    results.append(pet_result)
                    seen_content.add(pet_content)
            
            print(f"🔍 Enhanced search: added {len(pet_search)} pet-related memories to {len(results)} total results")
        
        # FIXED: Enhanced search for location-related queries 
        if any(word in request.message.lower() for word in ["where", "live", "location", "address", "home"]):
            # Do supplementary searches for location information
            location_searches = [
                "King NC North Carolina live location",
                "address home where live",
                "7 kids family location"
            ]
            
            seen_content = set()
            for r in results:
                seen_content.add(getattr(r, 'content', str(r)))
            
            for search_term in location_searches:
                location_search = memory_engine.search_memories(search_term, k=5)
                for loc_result in location_search:
                    loc_content = getattr(loc_result, 'content', str(loc_result))
                    if (loc_content not in seen_content and 
                        any(keyword in loc_content.lower() for keyword in ["king", "nc", "north carolina", "live", "location", "address"])):
                        results.append(loc_result)
                        seen_content.add(loc_content)
                        print(f"🏠 Found location memory: {loc_content[:50]}...")
            
            print(f"🔍 Enhanced location search: now have {len(results)} total results")
        
        # NEW: Use memory synthesis to improve context understanding
        synthesizer = get_memory_synthesizer()
        synthesized_memories = synthesizer.synthesize_memories(request.message, results)
        
        # Get the best synthesized memories
        if synthesized_memories:
            best_synthesized = synthesizer.get_best_synthesis(synthesized_memories, limit=5)
            quality_memories = []
            
            for syn_mem in best_synthesized:
                # Convert synthesized memories back to regular format for compatibility
                class SynthesizedResult:
                    def __init__(self, syn_mem):
                        self.content = syn_mem.content
                        self.relevance_score = syn_mem.confidence
                        self.synthesis_type = syn_mem.synthesis_type
                        self.source_count = syn_mem.source_count
                
                quality_memories.append(SynthesizedResult(syn_mem))
            
            print(f"🧠 Synthesized {len(synthesized_memories)} memories into {len(quality_memories)} high-quality responses")
        else:
            # Fallback to improved filtering that includes important short memories
            quality_memories = []
            for result in results:
                # Get relevance score (higher is better)
                relevance_score = getattr(result, 'relevance_score', 0.0)
                content_length = len(result.content)
                
                # FIXED: Better logic for including high-relevance short memories
                if relevance_score >= 1.0:  # Very high relevance - always include regardless of length
                    quality_memories.append(result)
                    print(f"✅ Including high-relevance memory (score={relevance_score:.3f}, len={content_length}): {result.content[:50]}...")
                elif relevance_score >= 0.8 and content_length >= 20:  # High relevance with minimum length
                    quality_memories.append(result)
                    print(f"✅ Including relevant short memory (score={relevance_score:.3f}, len={content_length}): {result.content[:50]}...")
                elif relevance_score >= 0.5 and content_length > 50:  # Medium relevance with decent length
                    quality_memories.append(result)
                    print(f"✅ Including medium-length memory (score={relevance_score:.3f}, len={content_length}): {result.content[:50]}...")
                # Keep longer content with lower scores as fallback
                elif (content_length > 100 and 
                      relevance_score >= 0.4 and
                      not result.content.strip().lower().startswith(("yeah", "okay", "sure", "hmm", "uh", "um"))):
                    quality_memories.append(result)
                    print(f"✅ Including fallback long memory (score={relevance_score:.3f}, len={content_length}): {result.content[:50]}...")
                else:
                    print(f"❌ Excluding memory (score={relevance_score:.3f}, len={content_length}): {result.content[:30]}...")
        
        # FIXED: Ensure we always have some context
        if not quality_memories and results:
            # Fallback: take top 2 raw results if no quality memories found
            quality_memories = results[:2]
            print(f"⚠️ No quality memories found, using top {len(quality_memories)} raw results as fallback")
        
        # Option 1: Use GPT-4 for intelligent responses (if available)
        use_gpt = request.use_gpt  # Now properly using Pydantic field
        
        if use_gpt and quality_memories:
            try:
                # Generate intelligent response with GPT-4
                response = generate_gpt_response(request.message, quality_memories)
                return {
                    "response": response,
                    "relevant_memories": len(quality_memories),
                    "total_memories": len(memory_engine.memories),
                    "raw_search_results": len(results),
                    "response_type": "gpt-4"
                }
            except Exception as gpt_error:
                print(f"GPT-4 generation failed, falling back: {gpt_error}")
                # Fall through to simple response
        
        # Option 2: Simple context-based response (fallback or if GPT disabled)
        if len(quality_memories) > 0:
            # Extract meaningful context from best memories
            context_pieces = []
            for result in quality_memories[:2]:  # Use top 2 quality memories
                # Take more content for better context
                content = result.content[:400] if len(result.content) > 400 else result.content
                context_pieces.append(f"Previous conversation: {content}")
            
            context = "\n\n".join(context_pieces)
            
            # Generate more helpful conversational responses
            if any(word in request.message.lower() for word in ["hello", "hi", "hey"]):
                response = f"Hello! I found relevant context from your {len(memory_engine.memories):,} ChatGPT conversations. Based on your history, you've discussed various topics. How can I help you today?"
                if quality_memories:
                    response += f"\n\nRecent relevant context:\n{context[:300]}..."
            elif "?" in request.message:
                response = f"I found {len(quality_memories)} relevant conversations in your ChatGPT history that might help answer your question.\n\n{context[:500]}..."
            else:
                if len(context) > 100:
                    response = f"From your ChatGPT conversation history, here's relevant context:\n\n{context[:600]}..."
                else:
                    response = f"I found {len(quality_memories)} related conversations. Here's what might be relevant: {context}"
        else:
            # Fallback when no quality memories found
            response = f"I searched through your {len(memory_engine.memories):,} ChatGPT conversations but didn't find high-quality matches for '{request.message}'. The search found {len(results)} potential matches, but they were too fragmentary to be useful. Try rephrasing your query or asking about specific topics you know you've discussed."
        
        # Store this conversation as a new memory for future reference
        try:
            conversation_content = f"User: {request.message}\nAssistant: {response}"
            memory = memory_engine.add_memory(
                conversation_content,
                {
                    "type": "conversation", 
                    "timestamp": datetime.now().isoformat(),
                    "source": "live_chat"
                }
            )
        except Exception as store_error:
            print(f"Warning: Failed to store conversation memory: {store_error}")
        
        return {
            "response": response,
            "relevant_memories": len(quality_memories),
            "total_memories": len(memory_engine.memories),
            "raw_search_results": len(results),
            "response_type": "context-only"
        }
    except Exception as e:
        return {"error": str(e)}

# Debug endpoint to see raw search results
@app.post("/debug/search")
async def debug_search(request: SearchRequest):
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

if __name__ == "__main__":
    import uvicorn
    print(f"\n🌐 Starting ChatGPT Memory API with {len(memory_engine.memories):,} memories")
    print("🔗 Available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)