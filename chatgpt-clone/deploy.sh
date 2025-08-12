#!/bin/bash

# Deploy ChatGPT Clone with AI Memory Layer
set -e

echo "🚀 Deploying ChatGPT Clone with AI Memory Layer..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Create data directory
mkdir -p data

# Build and start services
echo "🏗️ Building Docker images..."
docker compose build

echo "📦 Starting services..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost:3000/health >/dev/null 2>&1; then
    echo "✅ Next.js app is running on http://localhost:3000"
else
    echo "⚠️ Next.js app may still be starting up..."
fi

if curl -f http://localhost:8001/health >/dev/null 2>&1; then
    echo "✅ Memory API is running on http://localhost:8001"
else
    echo "⚠️ Memory API may still be starting up..."
fi

echo ""
echo "🎉 Deployment complete!"
echo "📱 ChatGPT Clone: http://localhost:3000"
echo "🧠 Memory API: http://localhost:8001"
echo "📊 Memory API docs: http://localhost:8001/docs"
echo ""
echo "To stop: docker compose down"
echo "To view logs: docker compose logs -f"