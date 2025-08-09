# AI Memory Layer Deployment Snapshots

This directory contains production-ready deployment snapshots of the AI Memory Layer system.

## Current Release

**v1.16.0 - Provider Autodiscovery + HTTP Server**
- **File**: `ai-memory-layer-v1.16.0-provider-autodiscovery-20250809-121415.zip`
- **Documentation**: `DEPLOYMENT_v1.16.0.md`
- **Size**: 526KB (code-only, no database files)
- **Status**: ✅ Production Ready

## Deployment Features

### 🔌 Plugin Architecture
- Provider autodiscovery with entry points
- Runtime registration with decorators
- Hot-reload capability for development

### 🌐 HTTP Server
- FastAPI REST API (`python -m memory_layer.server`)
- Endpoints: build, search, providers, health
- Production-ready with intelligent ranking

### 🧠 Human-like Intelligence
- Semantic×temporal×salience blending
- 7-day temporal decay with recency boost
- Critical content salience multipliers

### 🏗️ Enterprise Architecture
- FAISS index manager with metadata checksums
- SQLite content-hash cache for efficiency
- Comprehensive testing and CI integration

## Quick Deploy

```bash
# Extract snapshot
unzip ai-memory-layer-v1.16.0-*.zip
cd ai-memory-layer/

# Configure
export OPENAI_API_KEY=your_key_here

# Deploy
python -m memory_layer.server  # HTTP server on :8080
# OR
python -m memory_layer.cli build --dir ./docs --provider openai
```

## Snapshot Format

Each deployment snapshot includes:
- **Source Code**: All `.py`, `.md`, `.txt`, `.yml` files
- **Tests**: Complete test suite for validation
- **Configuration**: CI workflows, requirements, settings
- **Documentation**: Setup guides and API references

**Excluded**: Database files, logs, virtual environments, git history

## Validation

Every snapshot is validated with:
- ✅ Syntax checks for all Python files
- ✅ Import tests for core modules
- ✅ CLI smoke tests with mock providers
- ✅ HTTP server endpoint testing
- ✅ Contract compliance for all providers

## Version History

| Version | Date | Features | Size | Status |
|---------|------|----------|------|--------|
| v1.16.0 | 2025-08-09 | Provider autodiscovery, HTTP server, intelligent ranking | 526KB | ✅ Active |

## Support

For deployment issues:
1. Check `DEPLOYMENT_v1.16.0.md` for detailed instructions
2. Run smoke tests to validate environment
3. Review logs for configuration issues
4. Verify API keys and provider availability