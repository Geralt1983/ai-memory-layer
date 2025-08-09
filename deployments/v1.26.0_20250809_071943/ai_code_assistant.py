#!/usr/bin/env python3
"""
AI Code Assistant with Memory Integration
==========================================

Your personal AI assistant that automatically stays updated with every GitHub commit.
Combines your existing AI Memory Layer with GPT-4 API for intelligent code discussions.

Features:
- Auto-processes GitHub webhook commits into memory
- Uses FAISS vector search for relevant context retrieval
- GPT-4 powered conversations with full project history
- Web interface for natural chat interactions
- Maintains conversation continuity across sessions
- Automatic code analysis and suggestions

Architecture:
GitHub → Webhook → Memory Layer → FAISS Index → GPT-4 Assistant → Web Chat

Usage:
    python ai_code_assistant.py --start-server
    python ai_code_assistant.py --process-commit <commit_file>
    python ai_code_assistant.py --chat "What did I change in the last commit?"
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Import your existing memory components
try:
    from core.memory_engine import MemoryEngine
    from core.memory import Memory
    from storage.faiss_store import FaissVectorStore
    from integrations.openai_embeddings import OpenAIEmbeddingProvider
    MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Memory components not available: {e}")
    MEMORY_AVAILABLE = False

# OpenAI for GPT-4 assistant
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("❌ OpenAI library not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

# Web interface
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    print("⚠️  Web interface not available. Install: pip install fastapi uvicorn")
    WEB_AVAILABLE = False

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SYNC_DIR = Path("./chatgpt-sync")
MEMORY_DIR = Path("./data")
ASSISTANT_LOG = Path("./ai_assistant.log")
MODEL = "gpt-4o"  # or gpt-4-turbo, gpt-3.5-turbo

@dataclass
class ConversationMessage:
    """Represents a message in the conversation"""
    role: str  # user, assistant, system
    content: str
    timestamp: str
    context_used: Optional[List[str]] = None  # Memory IDs that influenced this response

@dataclass
class CommitContext:
    """Commit information for AI context"""
    sha: str
    author: str
    message: str
    timestamp: str
    files_changed: List[str]
    content: str  # Full markdown content

class AICodeAssistant:
    """AI Assistant with integrated memory and commit awareness"""
    
    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required")
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.conversation_history: List[ConversationMessage] = []
        
        # Initialize memory engine if available
        self.memory_engine = None
        if MEMORY_AVAILABLE:
            try:
                self.memory_engine = self._initialize_memory_engine()
                print("✅ Memory engine initialized")
            except Exception as e:
                print(f"⚠️  Memory engine initialization failed: {e}")
        
        # Load conversation history
        self._load_conversation_history()
        
        print(f"🤖 AI Code Assistant initialized")
        print(f"   Model: {MODEL}")
        print(f"   Memory: {'✅ Active' if self.memory_engine else '❌ Disabled'}")
    
    def _initialize_memory_engine(self) -> Optional[MemoryEngine]:
        """Initialize the memory engine with existing components"""
        try:
            # Use existing memory configuration
            MEMORY_DIR.mkdir(exist_ok=True)
            
            # Initialize embedding provider
            embedding_provider = OpenAIEmbeddingProvider()
            
            # Initialize vector store
            vector_store = FaissVectorStore(
                dimension=embedding_provider.dimension,
                persist_directory=str(MEMORY_DIR)
            )
            
            # Create memory engine
            memory_engine = MemoryEngine(
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                persist_path=str(MEMORY_DIR / "assistant_memories.json")
            )
            
            return memory_engine
            
        except Exception as e:
            print(f"❌ Failed to initialize memory engine: {e}")
            return None
    
    def _load_conversation_history(self):
        """Load previous conversation history"""
        history_file = Path("conversation_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.conversation_history = [
                        ConversationMessage(**msg) for msg in data
                    ]
                print(f"📚 Loaded {len(self.conversation_history)} previous messages")
            except Exception as e:
                print(f"⚠️  Failed to load conversation history: {e}")
    
    def _save_conversation_history(self):
        """Save conversation history"""
        try:
            with open("conversation_history.json", 'w') as f:
                json.dump([asdict(msg) for msg in self.conversation_history], f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save conversation history: {e}")
    
    def process_commit_file(self, commit_file: Path) -> CommitContext:
        """Process a webhook-generated commit file into memory"""
        try:
            content = commit_file.read_text()
            
            # Parse commit info from markdown
            commit_info = self._parse_commit_markdown(content)
            
            # Store in memory if available
            if self.memory_engine:
                memory = Memory(
                    content=content,
                    metadata={
                        "type": "commit",
                        "sha": commit_info.sha,
                        "author": commit_info.author,
                        "timestamp": commit_info.timestamp,
                        "files_changed": len(commit_info.files_changed),
                        "source": "github_webhook"
                    }
                )
                
                self.memory_engine.add_memory(memory)
                print(f"💾 Stored commit {commit_info.sha[:8]} in memory")
            
            return commit_info
            
        except Exception as e:
            print(f"❌ Failed to process commit file: {e}")
            raise
    
    def _parse_commit_markdown(self, content: str) -> CommitContext:
        """Parse commit information from webhook-generated markdown"""
        lines = content.split('\n')
        
        sha = "unknown"
        author = "unknown"  
        message = "unknown"
        timestamp = datetime.now().isoformat()
        files_changed = []
        
        for line in lines:
            if '**SHA**:' in line and '`' in line:
                sha = line.split('`')[1]
            elif '**Author**:' in line:
                author = line.split(': ', 1)[1] if ': ' in line else "unknown"
            elif '**Message**:' in line:
                message = line.split(': ', 1)[1] if ': ' in line else "unknown"
            elif '**Time**:' in line:
                timestamp = line.split(': ', 1)[1] if ': ' in line else timestamp
            elif line.startswith('### ') and '`' in line:
                # Extract filename from markdown headers like "### ➕ `filename.py` (added)"
                filename = line.split('`')[1] if '`' in line else ""
                if filename:
                    files_changed.append(filename)
        
        return CommitContext(
            sha=sha,
            author=author,
            message=message,
            timestamp=timestamp,
            files_changed=files_changed,
            content=content
        )
    
    def get_relevant_context(self, query: str, max_results: int = 5) -> List[Memory]:
        """Retrieve relevant memories for the query"""
        if not self.memory_engine:
            return []
        
        try:
            return self.memory_engine.search_memories(query, max_results=max_results)
        except Exception as e:
            print(f"⚠️  Memory search failed: {e}")
            return []
    
    def chat(self, user_message: str) -> str:
        """Have a conversation with the AI assistant"""
        try:
            # Get relevant context from memory
            relevant_memories = self.get_relevant_context(user_message)
            context_used = [mem.id for mem in relevant_memories] if relevant_memories else None
            
            # Build context for GPT-4
            system_message = self._build_system_message(relevant_memories)
            
            # Prepare messages for OpenAI API
            messages = [{"role": "system", "content": system_message}]
            
            # Add recent conversation history (last 10 messages)
            recent_messages = self.conversation_history[-10:] if self.conversation_history else []
            for msg in recent_messages:
                if msg.role != "system":  # Don't include system messages in history
                    messages.append({"role": msg.role, "content": msg.content})
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Call GPT-4
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            assistant_message = response.choices[0].message.content
            
            # Save conversation
            self.conversation_history.append(ConversationMessage(
                role="user",
                content=user_message,
                timestamp=datetime.now().isoformat(),
                context_used=context_used
            ))
            
            self.conversation_history.append(ConversationMessage(
                role="assistant",
                content=assistant_message,
                timestamp=datetime.now().isoformat(),
                context_used=context_used
            ))
            
            self._save_conversation_history()
            
            return assistant_message
            
        except Exception as e:
            error_msg = f"❌ Chat error: {e}"
            print(error_msg)
            return error_msg
    
    def _build_system_message(self, relevant_memories: List[Memory]) -> str:
        """Build system message with project context"""
        base_prompt = """You are an AI Code Assistant for the AI Memory Layer project. You help with code review, debugging, architecture decisions, and project development.

Project Context:
- This is a Python-based AI memory system using FAISS for vector storage
- Uses OpenAI embeddings and ChatGPT integration
- Has FastAPI REST API and GitHub webhook integration
- Includes comprehensive testing and deployment automation

Your capabilities:
- Analyze code changes and provide feedback
- Suggest improvements and catch potential issues
- Help with architecture and design decisions
- Answer questions about the codebase
- Provide debugging assistance

Communication style:
- Be concise but thorough
- Use technical language appropriately
- Provide code examples when helpful
- Ask clarifying questions when needed"""

        if relevant_memories:
            context_section = "\n\nRelevant Project Context:\n"
            for i, memory in enumerate(relevant_memories, 1):
                preview = memory.content[:200] + "..." if len(memory.content) > 200 else memory.content
                context_section += f"\n{i}. {preview}"
                if memory.metadata:
                    if memory.metadata.get('type') == 'commit':
                        context_section += f" [Commit: {memory.metadata.get('sha', 'unknown')[:8]}]"
            
            base_prompt += context_section
        
        return base_prompt
    
    def process_all_commits(self) -> int:
        """Process all commit files in sync directory"""
        if not SYNC_DIR.exists():
            print(f"❌ Sync directory not found: {SYNC_DIR}")
            return 0
        
        md_files = list(SYNC_DIR.glob("*.md"))
        if not md_files:
            print("❌ No commit files found")
            return 0
        
        processed = 0
        for md_file in sorted(md_files, key=lambda f: f.stat().st_mtime):
            try:
                self.process_commit_file(md_file)
                processed += 1
                print(f"✅ Processed: {md_file.name}")
            except Exception as e:
                print(f"❌ Failed to process {md_file.name}: {e}")
        
        return processed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get assistant statistics"""
        stats = {
            "conversation_messages": len(self.conversation_history),
            "model": MODEL,
            "memory_enabled": self.memory_engine is not None,
            "openai_configured": bool(OPENAI_API_KEY)
        }
        
        if self.memory_engine:
            try:
                memories = self.memory_engine.get_all_memories()
                commit_memories = [m for m in memories if m.metadata.get('type') == 'commit']
                stats.update({
                    "total_memories": len(memories),
                    "commit_memories": len(commit_memories),
                    "memory_engine_type": type(self.memory_engine).__name__
                })
            except Exception as e:
                stats["memory_error"] = str(e)
        
        return stats

# Web Interface (if available)
if WEB_AVAILABLE:
    app = FastAPI(title="AI Code Assistant", version="1.0.0")
    
    # Global assistant instance
    assistant = None
    
    class ChatRequest(BaseModel):
        message: str
    
    class ChatResponse(BaseModel):
        response: str
        context_used: Optional[List[str]] = None
    
    @app.on_event("startup")
    async def startup():
        global assistant
        try:
            assistant = AICodeAssistant()
        except Exception as e:
            print(f"❌ Failed to initialize assistant: {e}")
    
    @app.get("/", response_class=HTMLResponse)
    async def web_interface():
        """Simple web chat interface"""
        html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Code Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background-color: #e3f2fd; text-align: right; }
        .assistant { background-color: #f3e5f5; }
        .input-container { display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 10px; }
        button { padding: 10px 20px; }
        .stats { background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🤖 AI Code Assistant</h1>
    <div class="stats" id="stats">Loading stats...</div>
    
    <div class="chat-container" id="chatContainer"></div>
    
    <div class="input-container">
        <input type="text" id="messageInput" placeholder="Ask about your code, commits, or project..." onkeypress="if(event.key==='Enter') sendMessage()">
        <button onclick="sendMessage()">Send</button>
    </div>
    
    <script>
        async function loadStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();
                document.getElementById('stats').innerHTML = 
                    `📊 Messages: ${stats.conversation_messages} | 💾 Memories: ${stats.total_memories || 0} | 🔄 Commits: ${stats.commit_memories || 0} | 🤖 Model: ${stats.model}`;
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage('user', message);
            input.value = '';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                
                const data = await response.json();
                addMessage('assistant', data.response);
                loadStats(); // Refresh stats
            } catch (e) {
                addMessage('assistant', 'Error: ' + e.message);
            }
        }
        
        function addMessage(role, content) {
            const container = document.getElementById('chatContainer');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `<strong>${role === 'user' ? 'You' : '🤖 Assistant'}:</strong><br>${content.replace(/\\n/g, '<br>')}`;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        // Load stats on page load
        loadStats();
    </script>
</body>
</html>
        '''
        return HTMLResponse(content=html_content)
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        if not assistant:
            raise HTTPException(status_code=503, detail="Assistant not initialized")
        
        response = assistant.chat(request.message)
        return ChatResponse(response=response)
    
    @app.get("/stats")
    async def stats_endpoint():
        if not assistant:
            raise HTTPException(status_code=503, detail="Assistant not initialized")
        
        return assistant.get_stats()
    
    @app.post("/process-commits")
    async def process_commits_endpoint():
        if not assistant:
            raise HTTPException(status_code=503, detail="Assistant not initialized")
        
        count = assistant.process_all_commits()
        return {"processed": count, "message": f"Processed {count} commit files"}

def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("AI Code Assistant")
        print("=" * 40)
        print("Usage:")
        print("  python ai_code_assistant.py --start-server")
        print("  python ai_code_assistant.py --process-commits")
        print("  python ai_code_assistant.py --chat 'What changed in the latest commit?'")
        print("  python ai_code_assistant.py --stats")
        return 1
    
    command = sys.argv[1]
    
    try:
        if command == "--start-server":
            if not WEB_AVAILABLE:
                print("❌ Web interface not available. Install: pip install fastapi uvicorn")
                return 1
            
            print("🚀 Starting AI Code Assistant server...")
            print("   Web interface: http://localhost:8002")
            uvicorn.run(app, host="0.0.0.0", port=8002)
            
        elif command == "--process-commits":
            assistant = AICodeAssistant()
            count = assistant.process_all_commits()
            print(f"✅ Processed {count} commit files")
            
        elif command == "--chat":
            if len(sys.argv) < 3:
                print("❌ Please provide a message")
                return 1
            
            assistant = AICodeAssistant()
            response = assistant.chat(sys.argv[2])
            print(f"\n🤖 Assistant: {response}")
            
        elif command == "--stats":
            assistant = AICodeAssistant()
            stats = assistant.get_stats()
            print("📊 AI Code Assistant Stats:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
        else:
            print(f"❌ Unknown command: {command}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())