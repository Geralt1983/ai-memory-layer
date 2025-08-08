#!/bin/bash
# 
# ChatGPT Memory System Deployment Script
# Deploys the complete 23,710 memory solution to EC2
#

set -e

# Configuration
EC2_HOST="ubuntu@18.224.179.36"
REMOTE_DIR="~/ai-memory-layer"
SERVICE_NAME="chatgpt-memory-api"

echo "🚀 Deploying ChatGPT Memory System to EC2..."

# Check if we can connect
echo "🔗 Testing connection to EC2..."
if ! ssh -o ConnectTimeout=10 $EC2_HOST "echo 'Connection successful'"; then
    echo "❌ Cannot connect to EC2 server"
    echo "💡 Make sure your SSH key is set up correctly"
    exit 1
fi

# Deploy the solution files
echo "📁 Syncing ChatGPT memory solution..."
rsync -avz --progress \
    rebuild_faiss_index.py \
    optimized_memory_loader.py \
    test_full_loader.py \
    run_chatgpt_api.py \
    fixed_direct_chatgpt_api.py \
    $EC2_HOST:$REMOTE_DIR/

# Create startup script on server
echo "⚙️ Creating ChatGPT memory startup script..."
ssh $EC2_HOST << 'REMOTE_SCRIPT'
cd ~/ai-memory-layer

# Create startup script for ChatGPT memory API
cat > start_chatgpt_memory.sh << 'STARTUP_EOF'
#!/bin/bash
set -e

cd ~/ai-memory-layer
source venv/bin/activate

echo "🚀 Starting ChatGPT Memory API..."
echo "📊 Data verification:"
ls -lh data/chatgpt_memories.json data/faiss_chatgpt_index.*

# Check if FAISS index needs rebuilding
if [ ! -f "data/faiss_chatgpt_index.index" ] || [ ! -f "data/faiss_chatgpt_index.pkl" ]; then
    echo "🔧 Building FAISS index (one-time setup)..."
    python rebuild_faiss_index.py
fi

# Test the loader before starting API
echo "🧪 Testing memory loader..."
python test_full_loader.py

if [ $? -eq 0 ]; then
    echo "✅ Memory loader test passed - starting API server..."
    # Use the optimized loader for production
    python optimized_memory_loader.py &
    SERVER_PID=$!
    
    echo "🌐 ChatGPT Memory API started with PID: $SERVER_PID"
    echo "📊 Ready to serve 23,710 ChatGPT memories"
    echo "🔗 API available at: http://18.224.179.36:8000"
    
    # Keep the server running
    wait $SERVER_PID
else
    echo "❌ Memory loader test failed - check logs"
    exit 1
fi
STARTUP_EOF

chmod +x start_chatgpt_memory.sh

# Create systemd service for ChatGPT memory API
sudo tee /etc/systemd/system/chatgpt-memory-api.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=ChatGPT Memory API with 23,710 conversations
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-memory-layer
Environment=PATH=/home/ubuntu/ai-memory-layer/venv/bin
EnvironmentFile=/home/ubuntu/ai-memory-layer/.env
ExecStart=/home/ubuntu/ai-memory-layer/start_chatgpt_memory.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Stop any existing API services to avoid conflicts
echo "🛑 Stopping existing services..."
sudo systemctl stop ai-code-assistant 2>/dev/null || true
pkill -f python || true
sleep 3

# Start the ChatGPT memory service
echo "🔧 Starting ChatGPT Memory API service..."
sudo systemctl daemon-reload
sudo systemctl enable chatgpt-memory-api
sudo systemctl restart chatgpt-memory-api

# Wait for service to initialize
echo "⏳ Waiting for service to initialize..."
sleep 15

# Check service status
echo "📊 Checking service status..."
sudo systemctl status chatgpt-memory-api --no-pager -l

echo "✅ ChatGPT Memory System deployment completed!"

REMOTE_SCRIPT

echo ""
echo "🎉 Deployment Complete!"
echo "🔗 ChatGPT Memory API: http://18.224.179.36:8000"
echo "📊 With 23,710 conversation memories loaded"
echo ""
echo "📋 Management commands:"
echo "  Status:  ssh $EC2_HOST 'sudo systemctl status chatgpt-memory-api'"
echo "  Logs:    ssh $EC2_HOST 'sudo journalctl -u chatgpt-memory-api -f'"
echo "  Restart: ssh $EC2_HOST 'sudo systemctl restart chatgpt-memory-api'"
echo "  Test:    ssh $EC2_HOST 'cd ~/ai-memory-layer && source venv/bin/activate && python test_full_loader.py'"