#!/usr/bin/env python3
"""
Private AI Assistant for AI Memory Layer Project
================================================

Your personal AI assistant that automatically stays updated with every GitHub commit.
Better than ChatGPT uploads - persistent memory, continuous context, automatic updates.

Features:
- Automatic commit processing from GitHub webhooks
- Persistent conversation memory with FAISS search
- GPT-4 powered with full project context
- Beautiful web chat interface
- Continuous architectural awareness
- No manual file uploads needed ever

Architecture:
GitHub → Webhook → Commit Processing → Memory Storage → GPT-4 → Web Chat

Usage:
    python private_ai_assistant.py --start
    python private_ai_assistant.py --process-repo
    python private_ai_assistant.py --chat "What changed in the latest commit?"
"""

import os
import json
import asyncio
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

# Core dependencies
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("❌ OpenAI required: pip install openai")
    OPENAI_AVAILABLE = False

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    VECTOR_AVAILABLE = True
except ImportError:
    print("⚠️  Vector search not available: pip install numpy scikit-learn")
    VECTOR_AVAILABLE = False

try:
    from fastapi import FastAPI, Request, HTTPException, WebSocket
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    print("⚠️  Web interface not available: pip install fastapi uvicorn websockets")
    WEB_AVAILABLE = False

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SYNC_DIR = Path("./chatgpt-sync")
DATA_DIR = Path("./assistant_data")
MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
PORT = 8003

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
logger = logging.getLogger("PrivateAssistant")

@dataclass
class Memory:
    """Simple memory structure with vector embedding"""
    id: str
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        return cls(**data)

@dataclass
class ChatMessage:
    """Chat message structure"""
    role: str  # user, assistant, system
    content: str
    timestamp: str
    memory_context: Optional[List[str]] = None

class MemoryStore:
    """Simple in-memory vector store with SQLite persistence"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.memories: Dict[str, Memory] = {}
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self._init_db()
        self._load_memories()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding TEXT,
                metadata TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def _load_memories(self):
        """Load memories from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('SELECT * FROM memories')
            
            for row in cursor:
                memory_id, content, embedding_json, metadata_json, timestamp = row
                
                embedding = json.loads(embedding_json) if embedding_json else None
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                memory = Memory(
                    id=memory_id,
                    content=content,
                    embedding=embedding,
                    metadata=metadata,
                    timestamp=timestamp
                )
                self.memories[memory_id] = memory
            
            conn.close()
            logger.info(f"Loaded {len(self.memories)} memories from database")
            
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def _save_memory(self, memory: Memory):
        """Save memory to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO memories (id, content, embedding, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                memory.id,
                memory.content,
                json.dumps(memory.embedding) if memory.embedding else None,
                json.dumps(memory.metadata),
                memory.timestamp
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save memory {memory.id}: {e}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using OpenAI API"""
        if not self.client:
            return None
        
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a new memory"""
        memory_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        embedding = self.get_embedding(content)
        
        memory = Memory(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat()
        )
        
        self.memories[memory_id] = memory
        self._save_memory(memory)
        
        logger.info(f"Added memory {memory_id}: {content[:50]}...")
        return memory_id
    
    def search_memories(self, query: str, max_results: int = 5) -> List[Memory]:
        """Search memories by semantic similarity"""
        if not VECTOR_AVAILABLE or not self.client:
            # Fallback to text search
            results = []
            query_lower = query.lower()
            for memory in self.memories.values():
                if query_lower in memory.content.lower():
                    results.append(memory)
            return results[:max_results]
        
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # Calculate similarities
            similarities = []
            for memory in self.memories.values():
                if memory.embedding:
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [memory.embedding]
                    )[0][0]
                    similarities.append((memory, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [mem for mem, _ in similarities[:max_results]]
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    def get_recent_memories(self, hours: int = 24, max_results: int = 10) -> List[Memory]:
        """Get recent memories"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = []
        for memory in self.memories.values():
            try:
                memory_time = datetime.fromisoformat(memory.timestamp.replace('Z', '+00:00'))
                if memory_time.replace(tzinfo=None) > cutoff:
                    recent.append(memory)
            except:
                continue
        
        recent.sort(key=lambda m: m.timestamp, reverse=True)
        return recent[:max_results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        commit_memories = [m for m in self.memories.values() if m.metadata.get('type') == 'commit']
        conversation_memories = [m for m in self.memories.values() if m.metadata.get('type') == 'conversation']
        
        return {
            "total_memories": len(self.memories),
            "commit_memories": len(commit_memories),
            "conversation_memories": len(conversation_memories),
            "recent_commits": len([m for m in commit_memories if 
                                 (datetime.now() - datetime.fromisoformat(m.timestamp.replace('Z', '+00:00').replace('+00:00', ''))).days < 7])
        }

class PrivateAIAssistant:
    """Your private AI assistant with persistent memory"""
    
    def __init__(self):
        if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.memory_store = MemoryStore(DATA_DIR / "memories.db")
        self.conversation_history: List[ChatMessage] = []
        
        # Load conversation history
        self._load_conversation_history()
        
        logger.info("🤖 Private AI Assistant initialized")
        logger.info(f"   Model: {MODEL}")
        logger.info(f"   Memories: {len(self.memory_store.memories)}")
    
    def _load_conversation_history(self):
        """Load conversation history from file"""
        history_file = DATA_DIR / "conversation_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.conversation_history = [ChatMessage(**msg) for msg in data]
                logger.info(f"Loaded {len(self.conversation_history)} conversation messages")
            except Exception as e:
                logger.error(f"Failed to load conversation history: {e}")
    
    def _save_conversation_history(self):
        """Save conversation history to file"""
        try:
            with open(DATA_DIR / "conversation_history.json", 'w') as f:
                json.dump([asdict(msg) for msg in self.conversation_history], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
    
    def process_commit_file(self, commit_file: Path) -> str:
        """Process a webhook-generated commit file"""
        try:
            content = commit_file.read_text()
            
            # Extract commit info
            commit_sha = "unknown"
            commit_author = "unknown"
            
            for line in content.split('\n'):
                if '**SHA**:' in line and '`' in line:
                    commit_sha = line.split('`')[1][:8]
                elif '**Author**:' in line:
                    commit_author = line.split(': ', 1)[1] if ': ' in line else "unknown"
            
            # Add to memory
            memory_id = self.memory_store.add_memory(
                content=content,
                metadata={
                    "type": "commit",
                    "sha": commit_sha,
                    "author": commit_author,
                    "source": "github_webhook",
                    "filename": commit_file.name
                }
            )
            
            logger.info(f"Processed commit {commit_sha} into memory {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to process commit file {commit_file}: {e}")
            raise
    
    def process_all_commits(self) -> int:
        """Process all commit files in sync directory"""
        if not SYNC_DIR.exists():
            logger.warning(f"Sync directory not found: {SYNC_DIR}")
            return 0
        
        md_files = list(SYNC_DIR.glob("*.md"))
        processed = 0
        
        for md_file in sorted(md_files, key=lambda f: f.stat().st_mtime):
            try:
                self.process_commit_file(md_file)
                processed += 1
            except Exception as e:
                logger.error(f"Failed to process {md_file.name}: {e}")
        
        logger.info(f"Processed {processed} commit files into memory")
        return processed
    
    def chat(self, user_message: str) -> str:
        """Have a conversation with the AI assistant"""
        try:
            # Search for relevant memories
            relevant_memories = self.memory_store.search_memories(user_message, max_results=5)
            memory_context = [mem.id for mem in relevant_memories]
            
            # Build system message with context
            system_message = self._build_system_message(relevant_memories)
            
            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": system_message}]
            
            # Add recent conversation history (last 10 messages)
            recent_messages = self.conversation_history[-10:]
            for msg in recent_messages:
                if msg.role != "system":
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
            
            assistant_response = response.choices[0].message.content
            
            # Save conversation
            self.conversation_history.append(ChatMessage(
                role="user",
                content=user_message,
                timestamp=datetime.now().isoformat(),
                memory_context=memory_context
            ))
            
            self.conversation_history.append(ChatMessage(
                role="assistant", 
                content=assistant_response,
                timestamp=datetime.now().isoformat(),
                memory_context=memory_context
            ))
            
            # Also save user question and assistant response as memories
            self.memory_store.add_memory(
                content=f"User: {user_message}\n\nAssistant: {assistant_response}",
                metadata={
                    "type": "conversation",
                    "user_query": user_message,
                    "context_memories": memory_context
                }
            )
            
            self._save_conversation_history()
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"❌ Chat error: {e}"
            logger.error(error_msg)
            return error_msg
    
    def _build_system_message(self, relevant_memories: List[Memory]) -> str:
        """Build system message with project context"""
        system_prompt = """You are a private AI assistant for the AI Memory Layer project. You are NOT ChatGPT - you are a custom, persistent assistant that maintains full memory of this project's evolution.

**Your Role:**
- Code review and analysis expert for the AI Memory Layer project
- Architecture advisor and debugging assistant  
- Persistent memory of all commits, conversations, and project evolution
- Direct, technical communication style

**Project Context:**
The AI Memory Layer is a Python-based system for maintaining conversation context using:
- FAISS vector storage for semantic search
- OpenAI embeddings and GPT-4 integration
- FastAPI REST API with GitHub webhook integration
- Comprehensive testing and automated deployment
- Now includes this private AI assistant (you!)

**Your Capabilities:**
- Full awareness of every commit and code change
- Semantic search through project history
- Architecture recommendations based on project evolution
- Debugging assistance with full context
- Code review with memory of previous decisions

**Communication Style:**
- Technical and direct
- Reference specific commits/changes when relevant
- Provide code examples and suggestions
- Ask clarifying questions when needed
- Remember and build on previous conversations"""

        if relevant_memories:
            context_section = "\n\n**Relevant Context from Memory:**\n"
            for i, memory in enumerate(relevant_memories, 1):
                preview = memory.content[:300] + "..." if len(memory.content) > 300 else memory.content
                context_section += f"\n{i}. {preview}"
                
                if memory.metadata.get('type') == 'commit':
                    sha = memory.metadata.get('sha', 'unknown')
                    context_section += f" [Commit: {sha}]"
                elif memory.metadata.get('type') == 'conversation':
                    context_section += " [Previous conversation]"
            
            system_prompt += context_section
        
        # Add recent project stats
        stats = self.memory_store.get_stats()
        system_prompt += f"\n\n**Current Project State:**\n"
        system_prompt += f"- Total memories: {stats['total_memories']}\n"
        system_prompt += f"- Commits tracked: {stats['commit_memories']}\n"
        system_prompt += f"- Conversations: {stats['conversation_memories']}\n"
        system_prompt += f"- Recent commits (7 days): {stats['recent_commits']}"
        
        return system_prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get assistant statistics"""
        memory_stats = self.memory_store.get_stats()
        
        return {
            "conversation_messages": len(self.conversation_history),
            "model": MODEL,
            "embedding_model": EMBEDDING_MODEL,
            **memory_stats
        }

# Web Interface
if WEB_AVAILABLE:
    app = FastAPI(title="Private AI Assistant", version="1.0.0")
    
    # Global assistant instance
    assistant: Optional[PrivateAIAssistant] = None
    
    class ChatRequest(BaseModel):
        message: str
    
    @app.on_event("startup")
    async def startup():
        global assistant
        try:
            assistant = PrivateAIAssistant()
            # Process any existing commits
            assistant.process_all_commits()
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {e}")
    
    @app.get("/", response_class=HTMLResponse)
    async def web_interface():
        """Beautiful web chat interface"""
        html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Private AI Assistant - AI Memory Layer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 15px 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header h1 {
            color: white;
            font-size: 24px;
            font-weight: 600;
        }
        
        .stats {
            color: rgba(255, 255, 255, 0.8);
            font-size: 14px;
            margin-top: 5px;
        }
        
        .main-container {
            flex: 1;
            display: flex;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
            gap: 20px;
            padding: 20px;
        }
        
        .chat-container {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
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
            align-items: flex-start;
            gap: 12px;
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 14px;
            flex-shrink: 0;
        }
        
        .user .message-avatar {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .assistant .message-avatar {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.4;
            word-wrap: break-word;
        }
        
        .user .message-content {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-bottom-right-radius: 6px;
        }
        
        .assistant .message-content {
            background: #f8f9fa;
            color: #333;
            border: 1px solid #e9ecef;
            border-bottom-left-radius: 6px;
        }
        
        .input-container {
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .input-wrapper {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        
        .message-input {
            flex: 1;
            padding: 12px 16px;
            border: none;
            border-radius: 25px;
            background: white;
            font-size: 16px;
            outline: none;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .send-button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
        }
        
        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .sidebar {
            width: 300px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        .sidebar h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
        }
        
        .sidebar-stats {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 14px;
        }
        
        .recent-activity {
            font-size: 14px;
            color: #666;
        }
        
        .activity-item {
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .loading {
            opacity: 0.6;
        }
        
        @media (max-width: 768px) {
            .main-container {
                flex-direction: column;
                padding: 10px;
            }
            
            .sidebar { width: 100%; }
            .message-content { max-width: 85%; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Private AI Assistant</h1>
        <div class="stats" id="headerStats">AI Memory Layer Project • Loading...</div>
    </div>
    
    <div class="main-container">
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="message-avatar">🤖</div>
                    <div class="message-content">
                        <strong>Hello! I'm your private AI assistant for the AI Memory Layer project.</strong><br><br>
                        I have persistent memory of all your commits, conversations, and code changes. Ask me anything about:
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>Recent commits and code changes</li>
                            <li>Architecture decisions and recommendations</li> 
                            <li>Debugging help with full project context</li>
                            <li>Code review and optimization suggestions</li>
                        </ul>
                        What would you like to know?
                    </div>
                </div>
            </div>
            
            <div class="input-container">
                <div class="input-wrapper">
                    <input type="text" class="message-input" id="messageInput" 
                           placeholder="Ask about commits, code, architecture..." 
                           onkeypress="if(event.key==='Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }">
                    <button class="send-button" id="sendButton" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <h3>📊 Assistant Stats</h3>
            <div class="sidebar-stats" id="sidebarStats">Loading...</div>
            
            <h3>🔄 Quick Actions</h3>
            <div class="recent-activity">
                <div class="activity-item">
                    <button onclick="askQuickQuestion('What changed in the latest commit?')" style="background: none; border: none; color: #667eea; cursor: pointer;">
                        📝 Latest commit changes
                    </button>
                </div>
                <div class="activity-item">
                    <button onclick="askQuickQuestion('What are the current architecture issues?')" style="background: none; border: none; color: #667eea; cursor: pointer;">
                        🏗️ Architecture review
                    </button>
                </div>
                <div class="activity-item">
                    <button onclick="askQuickQuestion('Show me recent performance improvements')" style="background: none; border: none; color: #667eea; cursor: pointer;">
                        ⚡ Performance analysis
                    </button>
                </div>
                <div class="activity-item">
                    <button onclick="processCommits()" style="background: none; border: none; color: #667eea; cursor: pointer;">
                        🔄 Process new commits
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let isLoading = false;
        
        async function loadStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();
                
                document.getElementById('headerStats').textContent = 
                    `${stats.total_memories} memories • ${stats.commit_memories} commits • ${stats.conversation_messages} messages`;
                
                document.getElementById('sidebarStats').innerHTML = `
                    <div class="stat-item"><span>💾 Total Memories:</span><span>${stats.total_memories}</span></div>
                    <div class="stat-item"><span>📝 Commits:</span><span>${stats.commit_memories}</span></div>
                    <div class="stat-item"><span>💬 Conversations:</span><span>${stats.conversation_memories}</span></div>
                    <div class="stat-item"><span>🔄 Recent Commits:</span><span>${stats.recent_commits}</span></div>
                    <div class="stat-item"><span>🤖 Model:</span><span>${stats.model}</span></div>
                `;
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message || isLoading) return;
            
            addMessage('user', message);
            input.value = '';
            isLoading = true;
            updateUI();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                
                const data = await response.json();
                addMessage('assistant', data.response);
                loadStats();
            } catch (e) {
                addMessage('assistant', '❌ Error: ' + e.message);
            } finally {
                isLoading = false;
                updateUI();
            }
        }
        
        function addMessage(role, content) {
            const container = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === 'user' ? 'You' : '🤖';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = content.replace(/\\n/g, '<br>');
            
            div.appendChild(avatar);
            div.appendChild(messageContent);
            container.appendChild(div);
            
            container.scrollTop = container.scrollHeight;
        }
        
        function askQuickQuestion(question) {
            document.getElementById('messageInput').value = question;
            sendMessage();
        }
        
        async function processCommits() {
            try {
                addMessage('assistant', '🔄 Processing new commits...');
                const response = await fetch('/process-commits', {method: 'POST'});
                const data = await response.json();
                addMessage('assistant', `✅ Processed ${data.processed} commits into memory.`);
                loadStats();
            } catch (e) {
                addMessage('assistant', '❌ Failed to process commits: ' + e.message);
            }
        }
        
        function updateUI() {
            const button = document.getElementById('sendButton');
            const input = document.getElementById('messageInput');
            
            button.disabled = isLoading;
            button.textContent = isLoading ? 'Thinking...' : 'Send';
            input.disabled = isLoading;
            
            if (isLoading) {
                document.body.classList.add('loading');
            } else {
                document.body.classList.remove('loading');
            }
        }
        
        // Load stats on page load
        loadStats();
        
        // Auto-refresh stats every 30 seconds
        setInterval(loadStats, 30000);
    </script>
</body>
</html>
        '''
        return HTMLResponse(content=html_content)
    
    @app.post("/chat")
    async def chat_endpoint(request: ChatRequest):
        if not assistant:
            raise HTTPException(status_code=503, detail="Assistant not initialized")
        
        response = assistant.chat(request.message)
        return {"response": response}
    
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
        return {"processed": count}

def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("🤖 Private AI Assistant for AI Memory Layer")
        print("=" * 50)
        print("Usage:")
        print("  python private_ai_assistant.py --start")
        print("  python private_ai_assistant.py --process-repo")
        print("  python private_ai_assistant.py --chat 'What changed recently?'")
        print("  python private_ai_assistant.py --stats")
        return 1
    
    command = sys.argv[1]
    
    try:
        if command == "--start":
            if not WEB_AVAILABLE:
                print("❌ Web interface not available: pip install fastapi uvicorn")
                return 1
            
            print("🚀 Starting Private AI Assistant...")
            print(f"   Web interface: http://localhost:{PORT}")
            print("   This is YOUR persistent AI assistant - better than ChatGPT uploads!")
            uvicorn.run(app, host="0.0.0.0", port=PORT)
            
        elif command == "--process-repo":
            assistant = PrivateAIAssistant()
            count = assistant.process_all_commits()
            print(f"✅ Processed {count} commits into persistent memory")
            
        elif command == "--chat":
            if len(sys.argv) < 3:
                print("❌ Please provide a message")
                return 1
            
            assistant = PrivateAIAssistant()
            response = assistant.chat(sys.argv[2])
            print(f"\n🤖 Assistant: {response}")
            
        elif command == "--stats":
            assistant = PrivateAIAssistant()
            stats = assistant.get_stats()
            print("📊 Private AI Assistant Stats:")
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