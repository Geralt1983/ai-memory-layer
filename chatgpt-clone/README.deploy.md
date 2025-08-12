# ChatGPT Clone - Deployment Guide

This ChatGPT clone integrates with the AI Memory Layer for context-aware conversations.

## Local Deployment

### Quick Start
```bash
# Clone and setup
git clone <repo>
cd chatgpt-clone

# Start with Docker
./deploy.sh
```

### Manual Setup
```bash
# Install dependencies
npm install
pip install -r requirements.txt

# Setup database
npx prisma migrate dev
npm run prisma:seed

# Start both services
npm run dev:full
# Or separately:
npm run dev          # Next.js on :3000
npm run dev:memory   # Python API on :8001
```

## Cloud Deployment Options

### 1. Railway (Recommended)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Environment Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key
- `DATABASE_URL` - PostgreSQL connection string (Railway provides this)
- `LOG_LEVEL` - INFO (default)
- `VECTOR_STORE_TYPE` - faiss

### 2. Docker + VPS
```bash
# Build and push to registry
docker build -t chatgpt-clone .
docker tag chatgpt-clone your-registry/chatgpt-clone
docker push your-registry/chatgpt-clone

# Deploy on VPS
docker run -d \
  -p 3000:3000 \
  -p 8001:8001 \
  -e OPENAI_API_KEY=your_key \
  -v ./data:/app/data \
  your-registry/chatgpt-clone
```

### 3. Vercel + Separate Python Service
Deploy Next.js to Vercel and Python API elsewhere:

```bash
# Deploy frontend to Vercel
npm install -g vercel
vercel

# Update memory API URL in environment
# Set MEMORY_API_URL in Vercel dashboard
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Next.js App   │    │  Python Memory  │
│   (Port 3000)   │◄──►│   API Service   │
│                 │    │   (Port 8001)   │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   SQLite/       │    │   FAISS Vector  │
│   PostgreSQL    │    │     Store       │
└─────────────────┘    └─────────────────┘
```

## Environment Variables

**Required:**
- `OPENAI_API_KEY` - OpenAI API key for real responses

**Optional:**
- `DATABASE_URL` - Database connection (defaults to SQLite)
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `LOG_FORMAT` - Log format (json, text)
- `VECTOR_STORE_TYPE` - Vector store type (faiss, chroma)
- `PERSIST_DIRECTORY` - Data persistence directory
- `MEMORY_PERSIST_PATH` - Memory file path

## Health Checks

- Next.js: `GET /health`
- Memory API: `GET /health` (port 8001)
- Combined: Both services report status

## Troubleshooting

**Memory API not starting:**
```bash
# Check Python dependencies
pip install -r requirements.txt

# Check if FAISS is available
python -c "import faiss; print('FAISS OK')"

# Check logs
docker compose logs memory-api
```

**Database issues:**
```bash
# Reset database
rm -rf data/dev.db
npx prisma migrate dev
```

**Port conflicts:**
```bash
# Check what's using ports
lsof -i :3000
lsof -i :8001

# Kill processes if needed
kill -9 <PID>
```