# AI Code Assistant

Intelligent assistant for the AI Memory Layer project with semantic search over commit history.

## Quick Start

```bash
# Setup
cd ai_code_assistant
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key-here"

# Index existing commits (one-time setup)
python main.py --index-commits

# Start the assistant
python main.py --start-server
```

Then open http://localhost:8001 in your browser.

## Features

- **Semantic Search**: Find relevant commits and code changes using natural language
- **GPT-4 Integration**: Get intelligent responses with full project context  
- **Memory Persistence**: FAISS vector storage with SQLite metadata
- **Web Interface**: Real-time chat with streaming responses
- **Auto Indexing**: Processes new commits automatically via webhook

## API Endpoints

- `POST /query` - Ask questions about the codebase
- `GET /memories` - List recent memories
- `GET /stats` - System statistics
- `GET /` - Web interface

## Architecture

1. **main.py** - FastAPI server and CLI commands
2. **embedder.py** - OpenAI embedding service with caching
3. **vector_store.py** - FAISS + SQLite storage
4. **memory_query.py** - Semantic search and ranking
5. **prompt_builder.py** - Context-aware prompt construction
6. **gpt_assistant.py** - GPT-4 conversation management

## Usage Examples

**CLI:**
```bash
# Index commits from parent directory
python main.py --index-commits --git-dir=../

# Start with custom port
python main.py --start-server --port=8080

# Query from command line
python main.py --query="How does the memory engine work?"
```

**Web Interface:**
- Ask: "What changed in the latest commit?"
- Ask: "How is the FAISS index optimized?"
- Ask: "Show me recent bug fixes"

**API:**
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the vector storage architecture"}'
```

The assistant has deep knowledge of your entire codebase and provides context-aware responses with commit references and code examples.