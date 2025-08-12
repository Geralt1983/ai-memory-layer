#!/bin/bash

# AI Code Assistant Deployment Script
# Deploys the AI assistant to EC2 and configures it for production

set -e

# Configuration
EC2_HOST="ubuntu@18.224.179.36"
EC2_KEY="~/.ssh/AI-memory.pem"
REMOTE_DIR="~/ai-memory-layer"
SERVICE_NAME="ai-code-assistant"
SERVICE_PORT=8001

echo "🚀 Deploying AI Code Assistant to EC2..."

# Sync AI assistant files to EC2
echo "📁 Syncing AI assistant files..."
rsync -avz --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    -e "ssh -i $EC2_KEY" \
    ./ai_code_assistant/ \
    $EC2_HOST:$REMOTE_DIR/ai_code_assistant/

# Deploy service configuration and scripts
echo "⚙️ Creating service configuration..."
ssh -i $EC2_KEY $EC2_HOST << 'REMOTE_COMMANDS'
cd ~/ai-memory-layer

# Create systemd service file for AI assistant
sudo tee /etc/systemd/system/ai-code-assistant.service > /dev/null << 'EOF'
[Unit]
Description=AI Code Assistant
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-memory-layer/ai_code_assistant
Environment=PATH=/home/ubuntu/ai-memory-layer/venv/bin
EnvironmentFile=/home/ubuntu/ai-memory-layer/.env
ExecStart=/home/ubuntu/ai-memory-layer/venv/bin/python main.py --start-server --host=0.0.0.0 --port=8001
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Install AI assistant dependencies
echo "📦 Installing AI assistant dependencies..."
source venv/bin/activate
cd ai_code_assistant
pip install -r requirements.txt

# Index existing commits for the AI assistant
echo "🔍 Indexing commits for AI assistant..."
python main.py --index-commits --git-dir=../

# Configure nginx for AI assistant
echo "🌐 Configuring nginx for AI assistant..."
sudo tee /etc/nginx/sites-available/ai-assistant > /dev/null << 'NGINX_EOF'
server {
    listen 8001;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
NGINX_EOF

# Enable nginx configuration
sudo ln -sf /etc/nginx/sites-available/ai-assistant /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Start and enable AI assistant service
echo "🔧 Starting AI assistant service..."
sudo systemctl daemon-reload
sudo systemctl enable ai-code-assistant
sudo systemctl restart ai-code-assistant

# Wait a moment for service to start
sleep 5

# Check service status
echo "✅ Checking AI assistant service status..."
sudo systemctl status ai-code-assistant --no-pager -l

# Test the service
echo "🧪 Testing AI assistant endpoint..."
curl -f http://localhost:8001/health || {
    echo "❌ Health check failed"
    sudo journalctl -u ai-code-assistant --no-pager -l
    exit 1
}

echo "✅ AI Code Assistant deployed successfully!"
echo "🌐 Available at: http://18.224.179.36:8001"
echo "📊 Health check: http://18.224.179.36:8001/health"
echo "💬 Web interface: http://18.224.179.36:8001/"

REMOTE_COMMANDS

echo "🎉 AI Code Assistant deployment completed!"
echo ""
echo "🔗 Access your AI assistant at:"
echo "   Web Interface: http://18.224.179.36:8001"
echo "   API Health: http://18.224.179.36:8001/health"
echo "   API Stats: http://18.224.179.36:8001/stats"
echo ""
echo "📋 Service management commands:"
echo "   Status: ssh $EC2_HOST 'sudo systemctl status ai-code-assistant'"
echo "   Logs: ssh $EC2_HOST 'sudo journalctl -u ai-code-assistant -f'"
echo "   Restart: ssh $EC2_HOST 'sudo systemctl restart ai-code-assistant'"