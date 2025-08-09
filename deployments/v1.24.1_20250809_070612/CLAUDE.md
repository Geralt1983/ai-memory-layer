# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Memory Layer - A Python-based system for maintaining conversation context and improving AI responses through vector storage and retrieval.

## Development Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Unix/macOS
pip install -r requirements.txt
pip install -e ".[dev]"   # For development with testing tools

# Configuration
cp .env.example .env      # Then add your OpenAI API key

# Code Quality
black .                   # Format code (88 char line length)
flake8 .                  # Lint code
mypy .                    # Type checking

# Running
python example.py         # Interactive demo
```

## Architecture

The system uses a modular plugin architecture with three main layers:

1. **Core Layer** (`core/`)
   - `MemoryEngine`: Central class managing memory storage and retrieval
   - `Memory` dataclass: Stores content, embeddings, metadata, and timestamps
   - `ContextBuilder`: Aggregates memories into prompts (4000 char default limit)

2. **Storage Layer** (`storage/`)
   - Abstract `VectorStore` interface in `core/memory_engine.py`
   - FAISS implementation: In-memory/file-based, L2 distance
   - ChromaDB implementation: Persistent document store

3. **Integration Layer** (`integrations/`)
   - `EmbeddingProvider` abstraction for different embedding models
   - OpenAI integration for chat completion and embeddings (ada-002)

## Key Implementation Notes

- **Compatibility Mode**: The system includes numpy-free fallback implementations for compatibility with environments where numpy installation hangs or fails. When numpy/faiss/chromadb are unavailable, the system gracefully falls back to basic functionality using mock implementations.
- **Memory Search**: Uses embedding similarity search when both vector store and embedding provider are available
- **Persistence**: Both vector stores and MemoryEngine support persistence:
  - Vector stores: FAISS saves to `.index/.pkl` files, ChromaDB has built-in persistence
  - MemoryEngine: Saves memory list to JSON file, auto-loads on initialization
- **Environment Variables**: Required in `.env`:
  - `OPENAI_API_KEY`
  - Optional: `VECTOR_STORE_TYPE` (faiss or chroma)
  - Optional: `PERSIST_DIRECTORY` (./data)
  - Optional: `MEMORY_PERSIST_PATH` (./data/memories.json)

## Testing

The project includes a comprehensive test suite using pytest:

```bash
# Install dev dependencies (includes pytest)
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest tests/test_memory_engine.py  # Specific test file
```

**Test Structure:**
- `tests/conftest.py` - Shared fixtures and mocks
- `tests/test_memory.py` - Memory class tests
- `tests/test_memory_engine.py` - Core engine tests with persistence
- `tests/test_storage.py` - FAISS and ChromaDB storage tests
- `tests/test_embeddings.py` - Embedding provider tests (mocked OpenAI)
- `tests/test_context_builder.py` - Context building tests
- `tests/test_openai_integration.py` - OpenAI integration tests (mocked)
- `tests/test_integration.py` - End-to-end integration tests

## REST API

The project includes a FastAPI-based REST API for remote access:

```bash
# Start API server
python run_api.py

# Or directly with uvicorn
uvicorn api.main:app --reload
```

**API Endpoints:**
- `GET /health` - Health check and status
- `POST /memories` - Create new memory
- `GET /memories` - Get recent memories
- `POST /memories/search` - Search memories by similarity
- `DELETE /memories` - Clear all memories
- `POST /chat` - Chat with AI using memory context
- `GET /stats` - Get memory statistics

**Documentation:**
- Interactive docs: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json

**Environment Variables for API:**
- All previous variables plus optional API-specific settings
- Server runs on port 8000 by default

## Logging

The project includes comprehensive structured logging:

**Configuration (via environment variables):**
- `LOG_LEVEL` - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
- `LOG_FORMAT` - json or text (default: json)
- `LOG_FILE` - Log file name (optional, console-only if not set)
- `LOG_DIR` - Log directory (default: ./logs)

**Features:**
- Structured JSON logging with consistent fields
- Performance monitoring with execution times
- API request/response logging with status codes
- Memory operation tracking
- OpenAI API call logging
- Automatic log rotation (10MB files, 5 backups)
- Contextual logging with operation metadata

**Usage:**
```python
from core.logging_config import get_logger, monitor_performance

logger = get_logger("my_module")
logger.info("Operation completed", extra={"count": 42})

@monitor_performance("my_operation")
def my_function():
    # Function will be automatically timed and logged
    pass
```

**Testing API:**
```bash
# Run API client example
python api_client_example.py

# Interactive testing
python api_client_example.py interactive
```

## Memory Management

The project includes comprehensive memory cleanup and archival strategies:

**Cleanup Strategies:**
- **Age-based**: Remove memories older than specified days
- **Size-based**: Keep only N most recent memories
- **Relevance-based**: Remove memories with low relevance scores
- **Duplicate detection**: Remove similar/duplicate content
- **Metadata-based**: Clean up by metadata criteria

**Archival Features:**
- **Compressed storage**: Gzip-compressed JSON archives
- **Tiered storage**: Hot/warm/cold memory management
- **Export functionality**: JSON, CSV, TXT formats
- **Archive restoration**: Restore memories from archives

**API Endpoints:**
- `GET /memories/stats` - Detailed memory statistics
- `POST /memories/cleanup` - Clean up memories with criteria
- `GET /archives` - List memory archives
- `POST /memories/export` - Export memories to file
- `POST /archives/{name}/restore` - Restore from archive

**Usage Examples:**
```bash
# Demo memory management features
python memory_management_demo.py

# Interactive demo
python memory_management_demo.py interactive

# API cleanup example
curl -X POST "http://localhost:8000/memories/cleanup" \
  -H "Content-Type: application/json" \
  -d '{"max_memories": 1000, "max_age_days": 90, "dry_run": true}'
```

## Common Tasks

- To add a new vector store: Implement the `VectorStore` abstract class
- To add a new embedding provider: Implement the `EmbeddingProvider` interface
- To modify context building: Update `ContextBuilder.build_context()`
- To add new API endpoints: Extend `api/main.py` and add models to `api/models.py`