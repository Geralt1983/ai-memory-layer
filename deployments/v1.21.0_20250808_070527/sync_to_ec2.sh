#!/bin/bash
#
# Sync AI Memory Layer to EC2 Instance
# Transfers the ChatGPT memory database and vector indexes to your server
#

# Configuration
EC2_HOST="your-ec2-instance.amazonaws.com"  # Replace with your EC2 hostname/IP
EC2_USER="ubuntu"  # Replace with your EC2 username
EC2_KEY="~/.ssh/your-key.pem"  # Replace with path to your EC2 key
EC2_PROJECT_DIR="/home/ubuntu/ai-memory-layer"  # Replace with your EC2 project path

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 AI Memory Layer EC2 Sync${NC}"
echo "========================================"

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo -e "${RED}❌ rsync is required but not installed.${NC}"
    exit 1
fi

# Function to sync with progress
sync_with_progress() {
    local source=$1
    local dest=$2
    local desc=$3
    
    echo -e "\n${YELLOW}📦 Syncing $desc...${NC}"
    rsync -avz --progress \
        -e "ssh -i $EC2_KEY" \
        "$source" \
        "$EC2_USER@$EC2_HOST:$dest"
}

# Create remote directories if needed
echo -e "\n${YELLOW}📁 Ensuring remote directories exist...${NC}"
ssh -i "$EC2_KEY" "$EC2_USER@$EC2_HOST" "mkdir -p $EC2_PROJECT_DIR/data"

# Sync main project files (excluding venv and large files)
echo -e "\n${YELLOW}📄 Syncing project files...${NC}"
rsync -avz --progress \
    -e "ssh -i $EC2_KEY" \
    --exclude 'venv/' \
    --exclude 'fresh_venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'logs/' \
    --exclude '*.log' \
    --exclude 'data/' \
    ./ \
    "$EC2_USER@$EC2_HOST:$EC2_PROJECT_DIR/"

# Sync ChatGPT memory data
echo -e "\n${GREEN}🧠 Syncing ChatGPT Memory Database...${NC}"
echo "This contains 23,234 imported conversations"

# Memory JSON file
sync_with_progress \
    "data/chatgpt_memories.json" \
    "$EC2_PROJECT_DIR/data/" \
    "ChatGPT memories (30MB)"

# FAISS vector indexes
sync_with_progress \
    "data/faiss_chatgpt_index.index" \
    "$EC2_PROJECT_DIR/data/" \
    "FAISS index (145MB)"

sync_with_progress \
    "data/faiss_chatgpt_index.pkl" \
    "$EC2_PROJECT_DIR/data/" \
    "FAISS metadata (317MB)"

# Sync other data files
echo -e "\n${YELLOW}📊 Syncing additional data files...${NC}"
rsync -avz \
    -e "ssh -i $EC2_KEY" \
    --include="*.json" \
    --exclude="chatgpt_memories.json" \
    data/*.json \
    "$EC2_USER@$EC2_HOST:$EC2_PROJECT_DIR/data/"

# Sync environment file (if exists)
if [ -f ".env" ]; then
    echo -e "\n${YELLOW}🔐 Syncing environment configuration...${NC}"
    scp -i "$EC2_KEY" .env "$EC2_USER@$EC2_HOST:$EC2_PROJECT_DIR/"
fi

echo -e "\n${GREEN}✅ Sync Complete!${NC}"
echo "========================================"
echo -e "${GREEN}📊 Summary:${NC}"
echo "  • 23,234 ChatGPT conversations synced"
echo "  • Vector embeddings ready for search"
echo "  • Total data transferred: ~490MB"
echo ""
echo -e "${YELLOW}🚀 Next steps on EC2:${NC}"
echo "  1. SSH to your instance: ssh -i $EC2_KEY $EC2_USER@$EC2_HOST"
echo "  2. cd $EC2_PROJECT_DIR"
echo "  3. Create virtual environment: python3 -m venv venv"
echo "  4. Activate it: source venv/bin/activate"
echo "  5. Install dependencies: pip install -r requirements.txt"
echo "  6. Run the API: python run_api.py"
echo ""
echo -e "${GREEN}Your AI Memory Layer is ready to deploy! 🎉${NC}"