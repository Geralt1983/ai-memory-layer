#!/bin/bash
# Restart the AI Memory Layer service on EC2

echo "🔄 Restarting AI Memory Layer service..."

# SSH to EC2 and restart the service
ssh -o StrictHostKeyChecking=no ubuntu@18.224.179.36 << 'EOF'
    cd ~/ai-memory-layer
    source venv/bin/activate
    
    echo "🛑 Stopping existing service..."
    sudo systemctl stop ai-memory-layer
    
    echo "📦 Loading new configuration..."
    sleep 2
    
    echo "🚀 Starting service with ChatGPT dataset..."
    sudo systemctl start ai-memory-layer
    
    echo "⏳ Waiting for service to initialize..."
    sleep 10
    
    echo "✅ Service status:"
    sudo systemctl status ai-memory-layer --no-pager
    
    echo "📊 Memory stats:"
    curl -s localhost:8000/stats | jq .
EOF