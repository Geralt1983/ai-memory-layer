# AI Memory Layer - Environment Configuration Template

**⚠️ Note: Sensitive values have been redacted for security**

## Main `.env` File Configuration

```env
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-[REDACTED-FOR-SECURITY]

# GPT-4o Model Configuration (Optimized for human-like responses)
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-ada-002
SYSTEM_PROMPT_PATH=./prompts/system_prompt_4o.txt

# Optional: Storage Configuration
VECTOR_STORE_TYPE=faiss  # Options: faiss, chroma
PERSIST_DIRECTORY=./data

# Optional: Memory Persistence
MEMORY_PERSIST_PATH=./data/memories.json

# Optional: Logging Configuration
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json  # json or text
LOG_FILE=ai_memory_layer.log  # Optional, if not set logs only to console
LOG_DIR=./logs
```

## Additional Environment Files

### `.env.chatgpt` (GitHub Sync Configuration)
```env
# ChatGPT GitHub Sync Configuration
GITHUB_TOKEN=github_pat_[REDACTED-FOR-SECURITY]
REPO=Geralt1983/ai-memory-layer
```

### `.env.webhook.example` (Webhook Configuration Template)
```env
# GitHub Webhook Configuration
WEBHOOK_SECRET=your_webhook_secret_here
GITHUB_TOKEN=your_github_token_here
CHATGPT_API_KEY=your_openai_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8001
SYNC_DIR=chatgpt-sync

# Optional: Security
ALLOWED_IPS=192.168.1.0/24,10.0.0.0/8
```

## Environment Variable Descriptions

### Core Configuration
- **OPENAI_API_KEY**: OpenAI API key for GPT-4 and embeddings
- **OPENAI_MODEL**: GPT model to use (currently gpt-4o)
- **EMBEDDING_MODEL**: Embedding model for vector storage

### Storage Configuration
- **VECTOR_STORE_TYPE**: Vector database type (faiss or chroma)
- **PERSIST_DIRECTORY**: Directory for persistent storage
- **MEMORY_PERSIST_PATH**: JSON file for memory persistence

### Logging Configuration
- **LOG_LEVEL**: Logging verbosity level
- **LOG_FORMAT**: Log output format (json or text)
- **LOG_FILE**: Optional log file name
- **LOG_DIR**: Directory for log files

### GitHub Integration
- **GITHUB_TOKEN**: GitHub personal access token
- **REPO**: Repository in format "username/repo-name"
- **WEBHOOK_SECRET**: Secret for webhook signature validation

## Project Architecture Context

The AI Memory Layer uses these environment variables to configure:

1. **OpenAI Integration**: GPT-4o for responses, text-embedding-ada-002 for vectors
2. **FAISS Vector Storage**: High-performance similarity search
3. **Memory Persistence**: JSON-based memory storage
4. **GitHub Automation**: Webhook integration for automatic updates
5. **Comprehensive Logging**: Structured JSON logging for monitoring

The system is designed to be production-ready with proper configuration management, security considerations, and scalable architecture.