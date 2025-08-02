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

- **Memory Search**: The `search_memories` method in MemoryEngine currently only returns recent memories - it needs to be fixed to use actual embedding similarity search
- **Persistence**: Vector stores support persistence but MemoryEngine doesn't persist its memory list
- **Environment Variables**: Required in `.env`:
  - `OPENAI_API_KEY`
  - `VECTOR_STORE_TYPE` (faiss or chroma)
  - `PERSIST_DIRECTORY` (./data)

## Testing

No test suite exists yet. When implementing tests:
- Use pytest (already in dev dependencies)
- Test both FAISS and ChromaDB backends
- Mock OpenAI API calls
- Test persistence and recovery

## Common Tasks

- To add a new vector store: Implement the `VectorStore` abstract class
- To add a new embedding provider: Implement the `EmbeddingProvider` interface
- To modify context building: Update `ContextBuilder.build_context()`