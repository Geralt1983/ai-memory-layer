#!/bin/bash
#
# Simple EC2 Sync Script
# Quick sync using standard SSH/SCP commands
#

# Get EC2 details from user
echo "🚀 AI Memory Layer EC2 Sync"
echo "============================"
echo ""
read -p "Enter your EC2 hostname/IP: " EC2_HOST
read -p "Enter your EC2 username (default: ubuntu): " EC2_USER
EC2_USER=${EC2_USER:-ubuntu}
read -p "Enter path to your SSH key: " EC2_KEY
read -p "Enter remote project directory (default: /home/$EC2_USER/ai-memory-layer): " EC2_DIR
EC2_DIR=${EC2_DIR:-/home/$EC2_USER/ai-memory-layer}

echo ""
echo "📋 Configuration:"
echo "  Host: $EC2_HOST"
echo "  User: $EC2_USER"
echo "  Key: $EC2_KEY"
echo "  Remote Dir: $EC2_DIR"
echo ""
read -p "Continue? (y/n): " confirm
if [[ $confirm != "y" ]]; then
    echo "Cancelled."
    exit 0
fi

# Create remote directory
echo ""
echo "📁 Creating remote directories..."
ssh -i "$EC2_KEY" "$EC2_USER@$EC2_HOST" "mkdir -p $EC2_DIR/data"

# Create a tar archive of essential files
echo ""
echo "📦 Creating archive of project files..."
tar -czf ai-memory-layer-sync.tar.gz \
    --exclude='venv' \
    --exclude='fresh_venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='logs' \
    --exclude='*.log' \
    --exclude='data/*.index' \
    --exclude='data/*.pkl' \
    --exclude='data/chatgpt_memories.json' \
    .

# Transfer the archive
echo ""
echo "📤 Transferring project files..."
scp -i "$EC2_KEY" ai-memory-layer-sync.tar.gz "$EC2_USER@$EC2_HOST:$EC2_DIR/"

# Extract on remote
echo ""
echo "📂 Extracting files on EC2..."
ssh -i "$EC2_KEY" "$EC2_USER@$EC2_HOST" "cd $EC2_DIR && tar -xzf ai-memory-layer-sync.tar.gz && rm ai-memory-layer-sync.tar.gz"

# Transfer large data files separately
echo ""
echo "🧠 Transferring ChatGPT memory database (this may take a few minutes)..."
echo "  • Memories JSON (30MB)"
scp -i "$EC2_KEY" data/chatgpt_memories.json "$EC2_USER@$EC2_HOST:$EC2_DIR/data/"

echo "  • FAISS index (145MB)"
scp -i "$EC2_KEY" data/faiss_chatgpt_index.index "$EC2_USER@$EC2_HOST:$EC2_DIR/data/"

echo "  • FAISS metadata (317MB)"
scp -i "$EC2_KEY" data/faiss_chatgpt_index.pkl "$EC2_USER@$EC2_HOST:$EC2_DIR/data/"

# Clean up local archive
rm -f ai-memory-layer-sync.tar.gz

echo ""
echo "✅ Sync Complete!"
echo ""
echo "📊 Next steps:"
echo "1. SSH to your EC2: ssh -i $EC2_KEY $EC2_USER@$EC2_HOST"
echo "2. cd $EC2_DIR"
echo "3. python3 -m venv venv && source venv/bin/activate"
echo "4. pip install -r requirements.txt"
echo "5. python run_api.py"
echo ""
echo "Your 23,234 ChatGPT conversations are now on EC2! 🎉"