#!/usr/bin/env python3
"""
Fix the root route for ngrok access
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv('.env.local')

# Import optimized loader
from optimized_memory_loader import create_optimized_chatgpt_engine

print("🚀 Loading ChatGPT Memory System...")
memory_engine = create_optimized_chatgpt_engine()
print(f"✅ {len(memory_engine.memories):,} memories loaded!")

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

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

# Root route - show API info as HTML
@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <html>
        <head><title>ChatGPT Memory API</title></head>
        <body style="font-family: Arial; margin: 40px; background: #f5f5f5;">
            <h1>🧠 ChatGPT Memory API</h1>
            <h2>📊 {len(memory_engine.memories):,} ChatGPT Conversations Loaded</h2>
            <h3>🔗 API Endpoints:</h3>
            <ul>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/stats">Statistics</a></li>
                <li><a href="/docs">API Documentation</a></li>
            </ul>
            <h3>💬 Try a search:</h3>
            <div style="margin: 20px 0;">
                <input type="text" id="searchQuery" placeholder="Search ChatGPT conversations..." style="width: 300px; padding: 8px;">
                <button onclick="searchMemories()" style="padding: 8px 15px;">Search</button>
                <div id="searchResults" style="margin-top: 20px;"></div>
            </div>
            
            <script>
                function searchMemories() {{
                    var query = document.getElementById('searchQuery').value;
                    if (!query) return;
                    
                    var resultsDiv = document.getElementById('searchResults');
                    resultsDiv.innerHTML = '🔍 Searching...';
                    
                    fetch('/memories/search', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ query: query, k: 5 }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.results && data.results.length > 0) {{
                            var html = '<h4>Found ' + data.results.length + ' results from ' + data.searched_memories.toLocaleString() + ' memories:</h4>';
                            data.results.forEach(function(result, i) {{
                                html += '<div style="border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px;">';
                                html += '<strong>Result ' + (i+1) + ':</strong> ' + result.content;
                                html += '<br><small>Relevance: ' + result.relevance_score.toFixed(3) + '</small>';
                                html += '</div>';
                            }});
                            resultsDiv.innerHTML = html;
                        }} else {{
                            resultsDiv.innerHTML = '<p>No results found.</p>';
                        }}
                    }})
                    .catch(error => {{
                        resultsDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                    }});
                }}
                
                // Allow Enter key to search
                document.getElementById('searchQuery').addEventListener('keypress', function(e) {{
                    if (e.key === 'Enter') {{
                        searchMemories();
                    }}
                }});
            </script>
            <p><em>This API serves all your ChatGPT conversation history with full-text search capabilities.</em></p>
        </body>
    </html>
    """

# All the API endpoints
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "memory_count": len(memory_engine.memories),
        "system": "chatgpt_memory_api_v2",
        "dataset_size": f"{len(memory_engine.memories):,} ChatGPT conversations"
    }

@app.get("/stats")
async def stats():
    return {
        "total_memories": len(memory_engine.memories),
        "faiss_vectors": memory_engine.vector_store.index.ntotal if hasattr(memory_engine.vector_store, 'index') else 0,
        "system_info": "optimized_chatgpt_loader",
        "data_source": "chatgpt_conversations",
        "ready": True
    }

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

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Search for relevant memories
        results = memory_engine.search_memories(request.message, k=5)
        context = f"Found {len(results)} relevant memories from {len(memory_engine.memories):,} ChatGPT conversations."
        
        return {
            "response": f"I searched your ChatGPT history: {context}",
            "relevant_memories": len(results),
            "total_memories": len(memory_engine.memories)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print(f"\n🌐 Starting ChatGPT Memory API with {len(memory_engine.memories):,} memories")
    print("🔗 Available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)