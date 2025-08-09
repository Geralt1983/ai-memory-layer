# AI Memory Layer

A modular persistent memory system for AI assistants with semantic search, vector storage, and intelligent conversation synthesis.

## Version

**v1.3.0** - Clean Architecture Refactor (2025-08-07)

## Features

- 🧠 **Persistent Memory**: 21,338 ChatGPT conversations with semantic search
- 🔍 **Semantic Search**: Relevance-based filtering with similarity thresholds  
- 🤖 **GPT-4 Synthesis**: Intelligent response generation using conversation history
- 💬 **Chat Interface**: Real-time metrics with professional web UI
- 🔄 **Memory Management**: Conversation threading and content quality filtering
- 🌐 **Modular API**: Clean FastAPI architecture with separated endpoints
- ☁️ **Stable Access**: Cloudflare Tunnel for persistent public URL

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Web Interface  │────▶│   REST API      │────▶│  Memory Engine  │
│  (ChatGPT-like) │     │  (FastAPI)      │     │  (Core Logic)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                ┌─────────────────────────┴─────────────────────────┐
                                │                                                   │
                        ┌───────▼────────┐                                 ┌────────▼────────┐
                        │ Vector Storage │                                 │ OpenAI          │
                        │ (FAISS/Chroma) │                                 │ Integration     │
                        └────────────────┘                                 └─────────────────┘
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/ai-memory-layer.git
cd ai-memory-layer

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# CLI Quick Start (New in v1.15.9)
export OPENAI_API_KEY=your_key_here  # or VOYAGE_API_KEY

# Build an index from a folder
python -m memory_layer.cli build --dir ./documents --provider openai

# Search the index
python -m memory_layer.cli search "your query here" --provider openai

# Start HTTP server
python -m memory_layer.server  # listens on localhost:8080
```

## HTTP Server API

```bash
# Build index via REST API
curl -X POST localhost:8080/build -H 'Content-Type: application/json' \
  -d '{"dir":"./samples"}'

# Search with human-like re-ranking (semantic×temporal×salience)
curl 'localhost:8080/search?q=code%20review&k=5'

# List available providers
curl 'localhost:8080/providers'

# Health check
curl 'localhost:8080/health'
```

# Run modular API (v1.3.0)
python api/main.py

# Or run legacy API
python api/run_optimized_api.py

# Set up public URL (optional)
integrations/cloudflare_tunnel.sh
```

## Live Demo

- **Public URL**: https://ethnic-eternal-effects-unwrap.trycloudflare.com
- **Status**: ✅ LIVE with 21,338 ChatGPT conversations
- **Features**: GPT-4 synthesis, semantic search, real-time metrics

## Production Deployment

Currently deployed on AWS EC2:
- **URL**: http://18.224.179.36
- **Instance**: t3.small (Ubuntu 22.04 LTS)
- **Service**: systemd (auto-restart enabled)
- **Web Server**: nginx

## API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/health` | GET | Health check and status |
| `/memories` | POST | Create new memory |
| `/memories` | GET | Get recent memories |
| `/memories/search` | POST | Search memories by similarity |
| `/chat` | POST | Chat with AI using memory context |
| `/memories/stats` | GET | Get memory statistics |
| `/conversations/generate-title` | POST | Generate conversation title |

## Configuration

Environment variables (`.env`):
```bash
OPENAI_API_KEY=your_key_here
VECTOR_STORE_TYPE=faiss  # or 'chroma'
PERSIST_DIRECTORY=./data
MEMORY_PERSIST_PATH=./data/memories.json
LOG_LEVEL=INFO
```

## Project Structure

```
ai-memory-layer/
├── api/                   # Modular FastAPI endpoints
│   ├── main.py           # Main API server (v1.3.0)
│   ├── run_optimized_api.py  # Legacy API server
│   └── endpoints/        # Separated endpoint modules
│       ├── chat.py       # Chat and GPT-4 synthesis
│       ├── memories.py   # Memory search and stats  
│       └── conversations.py  # Title generation
├── core/                  # Core memory engine logic
│   ├── memory_engine.py  # Main memory management
│   ├── gpt_response.py   # GPT-4 integration
│   ├── similarity_utils.py  # Relevance scoring
│   └── memory_chunking.py   # Conversation threading
├── static/               # Web interface assets
│   └── web_interface.html  # Professional chat UI
├── integrations/         # External service integrations
│   └── cloudflare_tunnel.sh  # Stable public URL
├── scripts/              # Data processing scripts
│   ├── thread_conversations.py  # Memory preprocessing
│   └── rebuild_cleaned_index.py  # Index optimization
├── prompts/              # Prompt templates
│   └── prompt_templates.md  # GPT-4 prompts
├── data/                 # Memory storage (gitignored)
│   ├── chatgpt_memories_cleaned.json
│   ├── faiss.index
│   └── faiss.pkl
├── tests/                # Test suite
├── .env.example         # Environment configuration
└── requirements.txt     # Python dependencies
```

## Development

```bash
# Run tests
pytest

# Code quality
black .
flake8 .
mypy .

# Run with hot reload
uvicorn api.main:app --reload
```

## Changelog

### v1.0.0 (2024-08-04)
- 🚀 Production deployment on AWS EC2
- ✨ Added ChatGPT-like web interface with conversation management
- 🔧 Fixed AI hallucination issues with memory context
- 🎨 Added message regeneration and copy buttons
- 🛠️ Implemented automatic conversation title generation
- 📦 Full systemd service configuration
- 🌐 nginx reverse proxy setup

### v0.9.0 (2024-08-03)
- Initial implementation of memory engine
- FAISS and ChromaDB storage backends
- OpenAI integration for embeddings and chat
- Basic REST API
- Memory cleanup and archival features

## License

MIT License - See LICENSE file for details

## Author

Jeremy - AI Memory Layer Project

---

**Deployed Version**: v1.8.9 | **Last Updated**: 2025-08-04