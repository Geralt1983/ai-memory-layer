#!/usr/bin/env python3
"""
AI Code Assistant - Main FastAPI Service
=========================================

Your personal AI code assistant with full project memory.
Connects to existing GitHub webhook pipeline for automatic commit processing.

Features:
- Automatic commit processing from webhook-generated .md files
- FAISS vector storage for semantic code search
- GPT-4 powered responses with full project context
- CLI and web interface for natural conversations
- Persistent memory across sessions

Usage:
    python main.py --start-server
    python main.py --query "What changed in the latest commit?"
    python main.py --index-commits  # Process existing commits
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# FastAPI and web components
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our components
from embedder import EmbeddingService
from vector_store import VectorStore
from memory_query import MemoryQuery
from prompt_builder import PromptBuilder
from gpt_assistant import GPTAssistant

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SYNC_DIR = Path(os.getenv("SYNC_DIR", "../chatgpt-sync"))
DATA_DIR = Path("./data")
PORT = int(os.getenv("AI_ASSISTANT_PORT", "8003"))

# Initialize directories
DATA_DIR.mkdir(exist_ok=True)
SYNC_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AICodeAssistant")

# FastAPI app
app = FastAPI(
    title="AI Code Assistant",
    description="Your personal AI assistant with full project memory",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
embedding_service: Optional[EmbeddingService] = None
vector_store: Optional[VectorStore] = None
memory_query: Optional[MemoryQuery] = None
prompt_builder: Optional[PromptBuilder] = None
gpt_assistant: Optional[GPTAssistant] = None

class QueryRequest(BaseModel):
    query: str
    max_context: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global embedding_service, vector_store, memory_query, prompt_builder, gpt_assistant
    
    logger.info("🚀 Starting AI Code Assistant")
    logger.info(f"   Sync directory: {SYNC_DIR}")
    logger.info(f"   Data directory: {DATA_DIR}")
    logger.info(f"   OpenAI configured: {bool(OPENAI_API_KEY)}")
    
    try:
        # Initialize components
        embedding_service = EmbeddingService(api_key=OPENAI_API_KEY)
        vector_store = VectorStore(data_dir=DATA_DIR)
        memory_query = MemoryQuery(vector_store=vector_store, embedding_service=embedding_service)
        prompt_builder = PromptBuilder()
        gpt_assistant = GPTAssistant(api_key=OPENAI_API_KEY)
        
        # Process any existing commits
        await process_existing_commits()
        
        logger.info("✅ AI Code Assistant ready")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise

async def process_existing_commits():
    """Process any existing commit files that haven't been indexed"""
    if not SYNC_DIR.exists():
        logger.warning(f"Sync directory not found: {SYNC_DIR}")
        return
    
    md_files = list(SYNC_DIR.glob("*.md"))
    processed = 0
    
    for md_file in sorted(md_files, key=lambda f: f.stat().st_mtime):
        try:
            await process_commit_file(md_file)
            processed += 1
        except Exception as e:
            logger.error(f"Failed to process {md_file.name}: {e}")
    
    if processed > 0:
        logger.info(f"📚 Processed {processed} existing commits")

async def process_commit_file(commit_file: Path):
    """Process a single commit file into the vector store"""
    try:
        content = commit_file.read_text()
        
        # Check if already processed
        file_hash = str(hash(content))
        if vector_store.is_processed(file_hash):
            return
        
        # Parse commit metadata
        metadata = parse_commit_metadata(content)
        metadata['source_file'] = commit_file.name
        metadata['file_hash'] = file_hash
        
        # Chunk the content for better embeddings
        chunks = chunk_commit_content(content)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            
            # Generate embedding and store
            embedding = await embedding_service.embed_text(chunk)
            vector_store.add_memory(
                content=chunk,
                embedding=embedding,
                metadata=chunk_metadata
            )
        
        # Mark as processed
        vector_store.mark_processed(file_hash)
        logger.info(f"✅ Processed {commit_file.name} ({len(chunks)} chunks)")
        
    except Exception as e:
        logger.error(f"Failed to process {commit_file}: {e}")
        raise

def parse_commit_metadata(content: str) -> Dict[str, Any]:
    """Extract metadata from commit markdown"""
    metadata = {
        "type": "commit",
        "sha": "unknown",
        "author": "unknown",
        "timestamp": datetime.now().isoformat(),
        "files_changed": []
    }
    
    lines = content.split('\n')
    for line in lines:
        if '**SHA**:' in line and '`' in line:
            metadata['sha'] = line.split('`')[1]
        elif '**Author**:' in line:
            metadata['author'] = line.split(': ', 1)[1] if ': ' in line else "unknown"
        elif '**Time**:' in line:
            metadata['timestamp'] = line.split(': ', 1)[1] if ': ' in line else metadata['timestamp']
        elif line.startswith('### ') and '`' in line:
            # Extract filename from headers like "### ➕ `filename.py` (added)"
            filename = line.split('`')[1] if '`' in line else ""
            if filename:
                metadata['files_changed'].append(filename)
    
    return metadata

def chunk_commit_content(content: str, max_chunk_size: int = 1000) -> List[str]:
    """Break commit content into semantic chunks"""
    chunks = []
    
    # Split by major sections
    sections = content.split('\n## ')
    
    for i, section in enumerate(sections):
        if i > 0:  # Add back the ## for non-first sections
            section = '## ' + section
        
        if len(section) <= max_chunk_size:
            chunks.append(section)
        else:
            # Further split large sections
            subsections = section.split('\n### ')
            for j, subsection in enumerate(subsections):
                if j > 0:
                    subsection = '### ' + subsection
                
                if len(subsection) <= max_chunk_size:
                    chunks.append(subsection)
                else:
                    # Split by paragraphs as last resort
                    paragraphs = subsection.split('\n\n')
                    current_chunk = ""
                    
                    for paragraph in paragraphs:
                        if len(current_chunk + paragraph) <= max_chunk_size:
                            current_chunk += paragraph + "\n\n"
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = paragraph + "\n\n"
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
    
    return [chunk.strip() for chunk in chunks if chunk.strip()]

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Beautiful web chat interface"""
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>🤖 AI Code Assistant</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            display: flex;
            gap: 20px;
            min-height: 100vh;
        }
        
        .chat-panel {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            max-height: 60vh;
        }
        
        .message {
            margin: 15px 0;
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 16px;
            flex-shrink: 0;
        }
        
        .user .avatar {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .assistant .avatar {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
        }
        
        .message-content {
            max-width: 75%;
            padding: 15px 20px;
            border-radius: 20px;
            line-height: 1.5;
            word-wrap: break-word;
        }
        
        .user .message-content {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-bottom-right-radius: 8px;
        }
        
        .assistant .message-content {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-bottom-left-radius: 8px;
        }
        
        .sources {
            margin-top: 10px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            font-size: 12px;
        }
        
        .source-item {
            margin: 5px 0;
            padding: 5px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #667eea;
        }
        
        .input-area {
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }
        
        .input-wrapper {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        
        .message-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            font-size: 16px;
            resize: vertical;
            min-height: 50px;
            max-height: 120px;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .message-input:focus {
            border-color: #667eea;
        }
        
        .send-btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            min-width: 80px;
        }
        
        .send-btn:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        
        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .sidebar {
            width: 320px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        .sidebar h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 24px;
            font-weight: 600;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        .quick-actions {
            margin-top: 20px;
        }
        
        .quick-btn {
            display: block;
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            background: #fff;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            text-align: left;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        }
        
        .quick-btn:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        
        .loading {
            opacity: 0.7;
            pointer-events: none;
        }
        
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
                padding: 10px;
            }
            
            .sidebar { width: 100%; order: -1; }
            .message-content { max-width: 90%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="chat-panel">
            <div class="header">
                <h1>🤖 AI Code Assistant</h1>
                <p>Your intelligent companion for the AI Memory Layer project</p>
            </div>
            
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="avatar">🤖</div>
                    <div class="message-content">
                        <strong>Hello! I'm your AI Code Assistant.</strong><br><br>
                        I have complete knowledge of your AI Memory Layer project, including:
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>🔄 All commit history and code changes</li>
                            <li>🏗️ Architecture patterns and decisions</li>
                            <li>🐛 Bug fixes and improvements</li>
                            <li>📚 Documentation and examples</li>
                        </ul>
                        Ask me anything about your code!
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-wrapper">
                    <textarea class="message-input" id="messageInput" 
                             placeholder="Ask about your code, commits, architecture..."
                             onkeypress="if(event.key==='Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }"></textarea>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <h3>📊 Project Stats</h3>
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card">
                    <div class="stat-number" id="totalMemories">-</div>
                    <div class="stat-label">Memories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="totalCommits">-</div>
                    <div class="stat-label">Commits</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="totalQueries">-</div>
                    <div class="stat-label">Queries</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="avgResponseTime">-</div>
                    <div class="stat-label">Avg Response</div>
                </div>
            </div>
            
            <div class="quick-actions">
                <h3>⚡ Quick Questions</h3>
                <button class="quick-btn" onclick="askQuestion('What changed in the latest commit?')">
                    📝 Latest commit changes
                </button>
                <button class="quick-btn" onclick="askQuestion('What are the main components of this system?')">
                    🏗️ System architecture
                </button>
                <button class="quick-btn" onclick="askQuestion('Show me recent performance improvements')">
                    ⚡ Performance updates
                </button>
                <button class="quick-btn" onclick="askQuestion('What tests should I write next?')">
                    🧪 Testing suggestions
                </button>
                <button class="quick-btn" onclick="askQuestion('How does the memory engine work?')">
                    🧠 Memory system deep dive
                </button>
                <button class="quick-btn" onclick="processCommits()">
                    🔄 Process new commits
                </button>
            </div>
        </div>
    </div>
    
    <script>
        let isLoading = false;
        let queryCount = 0;
        let totalResponseTime = 0;
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message || isLoading) return;
            
            addMessage('user', message);
            input.value = '';
            setLoading(true);
            
            const startTime = Date.now();
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: message})
                });
                
                const data = await response.json();
                const responseTime = Date.now() - startTime;
                
                addMessage('assistant', data.response, data.sources);
                
                // Update stats
                queryCount++;
                totalResponseTime += responseTime;
                updateStats();
                
            } catch (e) {
                addMessage('assistant', '❌ Error: ' + e.message);
            } finally {
                setLoading(false);
            }
        }
        
        function addMessage(role, content, sources = null) {
            const container = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = role === 'user' ? 'You' : '🤖';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = content.replace(/\\n/g, '<br>').replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');
            
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'sources';
                sourcesDiv.innerHTML = '<strong>📚 Sources:</strong>';
                
                sources.forEach(source => {
                    const sourceItem = document.createElement('div');
                    sourceItem.className = 'source-item';
                    sourceItem.innerHTML = `<strong>${source.metadata?.sha || 'Unknown'}:</strong> ${source.content.substring(0, 100)}...`;
                    sourcesDiv.appendChild(sourceItem);
                });
                
                messageContent.appendChild(sourcesDiv);
            }
            
            div.appendChild(avatar);
            div.appendChild(messageContent);
            container.appendChild(div);
            
            container.scrollTop = container.scrollHeight;
        }
        
        function askQuestion(question) {
            document.getElementById('messageInput').value = question;
            sendMessage();
        }
        
        function setLoading(loading) {
            isLoading = loading;
            const btn = document.getElementById('sendBtn');
            const input = document.getElementById('messageInput');
            
            btn.disabled = loading;
            btn.textContent = loading ? 'Thinking...' : 'Send';
            input.disabled = loading;
            
            document.body.classList.toggle('loading', loading);
        }
        
        async function loadStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();
                
                document.getElementById('totalMemories').textContent = stats.total_memories || 0;
                document.getElementById('totalCommits').textContent = stats.commit_memories || 0;
                document.getElementById('totalQueries').textContent = queryCount;
                document.getElementById('avgResponseTime').textContent = 
                    queryCount > 0 ? Math.round(totalResponseTime / queryCount) + 'ms' : '-';
                
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }
        
        function updateStats() {
            document.getElementById('totalQueries').textContent = queryCount;
            document.getElementById('avgResponseTime').textContent = 
                queryCount > 0 ? Math.round(totalResponseTime / queryCount) + 'ms' : '-';
        }
        
        async function processCommits() {
            try {
                addMessage('assistant', '🔄 Processing new commits from webhook...');
                const response = await fetch('/process-commits', {method: 'POST'});
                const data = await response.json();
                addMessage('assistant', `✅ Processed ${data.processed} commits. Ready for questions!`);
                loadStats();
            } catch (e) {
                addMessage('assistant', '❌ Failed to process commits: ' + e.message);
            }
        }
        
        // Load stats on startup and refresh periodically
        loadStats();
        setInterval(loadStats, 30000);
    </script>
</body>
</html>
    '''
    return HTMLResponse(content=html_content)

@app.post("/query", response_model=QueryResponse)
async def query_assistant(request: QueryRequest):
    """Main query endpoint for the AI assistant"""
    if not all([memory_query, prompt_builder, gpt_assistant]):
        raise HTTPException(status_code=503, detail="Assistant components not initialized")
    
    start_time = datetime.now()
    
    try:
        # Search for relevant memories
        relevant_memories = await memory_query.search(request.query, max_results=request.max_context)
        
        # Build prompt with context
        prompt = prompt_builder.build_prompt(
            query=request.query,
            memories=relevant_memories
        )
        
        # Get GPT response
        response = await gpt_assistant.chat(prompt)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Format sources
        sources = [
            {
                "content": mem.content[:500] + "..." if len(mem.content) > 500 else mem.content,
                "metadata": mem.metadata,
                "similarity": mem.similarity if hasattr(mem, 'similarity') else None
            }
            for mem in relevant_memories
        ]
        
        logger.info(f"Query processed in {processing_time:.2f}s: {request.query[:50]}...")
        
        return QueryResponse(
            response=response,
            sources=sources,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = {
        "status": "operational",
        "sync_dir": str(SYNC_DIR),
        "data_dir": str(DATA_DIR),
        "openai_configured": bool(OPENAI_API_KEY)
    }
    
    if vector_store:
        store_stats = vector_store.get_stats()
        stats.update(store_stats)
    
    return stats

@app.post("/process-commits")
async def process_commits_endpoint():
    """Process new commits from sync directory"""
    try:
        await process_existing_commits()
        stats = vector_store.get_stats() if vector_store else {}
        return {
            "processed": stats.get("total_memories", 0),
            "message": "Commits processed successfully"
        }
    except Exception as e:
        logger.error(f"Failed to process commits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook-notify")
async def webhook_notify(background_tasks: BackgroundTasks):
    """Endpoint for webhook to notify of new commits"""
    background_tasks.add_task(process_existing_commits)
    return {"message": "Processing new commits in background"}

def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("🤖 AI Code Assistant")
        print("=" * 40)
        print("Usage:")
        print("  python main.py --start-server")
        print("  python main.py --query 'What changed recently?'")
        print("  python main.py --index-commits")
        print("  python main.py --stats")
        return 1
    
    command = sys.argv[1]
    
    if command == "--start-server":
        print("🚀 Starting AI Code Assistant server...")
        print(f"   Web interface: http://localhost:{PORT}")
        print("   Your personal AI assistant with full project memory!")
        uvicorn.run(app, host="0.0.0.0", port=PORT)
        
    elif command == "--query":
        if len(sys.argv) < 3:
            print("❌ Please provide a query")
            return 1
        
        # CLI query mode
        import asyncio
        async def cli_query():
            # Initialize components
            global embedding_service, vector_store, memory_query, prompt_builder, gpt_assistant
            embedding_service = EmbeddingService(api_key=OPENAI_API_KEY)
            vector_store = VectorStore(data_dir=DATA_DIR)
            memory_query = MemoryQuery(vector_store=vector_store, embedding_service=embedding_service)
            prompt_builder = PromptBuilder()
            gpt_assistant = GPTAssistant(api_key=OPENAI_API_KEY)
            
            query = sys.argv[2]
            relevant_memories = await memory_query.search(query, max_results=5)
            prompt = prompt_builder.build_prompt(query=query, memories=relevant_memories)
            response = await gpt_assistant.chat(prompt)
            
            print(f"\n🤖 Assistant: {response}")
            
            if relevant_memories:
                print(f"\n📚 Based on {len(relevant_memories)} relevant memories:")
                for i, mem in enumerate(relevant_memories, 1):
                    sha = mem.metadata.get('sha', 'unknown')
                    print(f"   {i}. Commit {sha[:8]}: {mem.content[:100]}...")
        
        asyncio.run(cli_query())
        
    elif command == "--index-commits":
        # Index existing commits
        import asyncio
        async def index_commits():
            global embedding_service, vector_store
            embedding_service = EmbeddingService(api_key=OPENAI_API_KEY)
            vector_store = VectorStore(data_dir=DATA_DIR)
            await process_existing_commits()
            stats = vector_store.get_stats()
            print(f"✅ Indexed commits. Total memories: {stats.get('total_memories', 0)}")
        
        asyncio.run(index_commits())
        
    elif command == "--stats":
        # Show stats
        vector_store = VectorStore(data_dir=DATA_DIR)
        stats = vector_store.get_stats()
        print("📊 AI Code Assistant Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    else:
        print(f"❌ Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())