# AI Memory Layer

A persistent memory system for AI assistants with semantic search, vector storage, and conversation management.

## Version

**v1.1.2** - Production Release (2024-08-04)

## Features

- 🧠 **Persistent Memory**: Conversations and context preserved across sessions
- 🔍 **Semantic Search**: Vector-based memory retrieval using FAISS/ChromaDB
- 💬 **Chat Interface**: ChatGPT-like web interface with conversation management
- 🔄 **Memory Management**: Automatic cleanup, archival, and export capabilities
- 🌐 **REST API**: Full-featured API for integration with other applications
- ☁️ **Cloud Ready**: Deployed on AWS EC2 for 24/7 availability

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

## Quick Start (Local Development)

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

# Run API server
python run_api.py

# Open web interface
open web_interface_enhanced.html
```

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
├── api/                    # FastAPI REST endpoints
├── core/                   # Core memory engine logic
├── integrations/          # OpenAI and embedding integrations
├── storage/               # Vector storage implementations
├── tests/                 # Test suite
├── web_interface_enhanced.html  # Web UI
├── run_api.py            # API server entry point
├── cli_interface.py      # Command-line interface
└── requirements.txt      # Python dependencies
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

**Deployed Version**: v1.1.2 | **Last Updated**: 2024-08-04