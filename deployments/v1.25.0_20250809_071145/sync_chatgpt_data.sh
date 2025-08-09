#!/bin/bash
#
# ChatGPT Data Sync to EC2
# Transfers your 23,234 conversations with embeddings
#

# EC2 Configuration (matching deploy.sh)
EC2_HOST="ubuntu@18.224.179.36"
EC2_KEY="~/.ssh/AI-memory.pem"
PROJECT_DIR="~/ai-memory-layer"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🧠 ChatGPT Memory Data Sync${NC}"
echo "========================================"
echo -e "${BLUE}📊 Data to sync:${NC}"
echo "  • 23,234 conversations"
echo "  • 471MB total (with embeddings)"
echo "  • Target: $EC2_HOST"
echo ""

# Check if key exists
if [ ! -f ~/.ssh/AI-memory.pem ]; then
    echo -e "${YELLOW}⚠️  SSH key not found at ~/.ssh/AI-memory.pem${NC}"
    echo "Please ensure your EC2 key is in the correct location."
    exit 1
fi

# Create data directory on EC2
echo -e "${YELLOW}📁 Preparing EC2 directories...${NC}"
ssh -i $EC2_KEY $EC2_HOST "mkdir -p $PROJECT_DIR/data"

# Function to transfer with progress
transfer_file() {
    local file=$1
    local desc=$2
    local size=$3
    
    echo -e "\n${YELLOW}📤 Transferring $desc ($size)...${NC}"
    scp -i $EC2_KEY "$file" "$EC2_HOST:$PROJECT_DIR/data/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $desc transferred successfully${NC}"
    else
        echo -e "${RED}✗ Failed to transfer $desc${NC}"
        exit 1
    fi
}

# Transfer the memory files
echo -e "\n${GREEN}🚀 Starting data transfer...${NC}"

# ChatGPT memories JSON
transfer_file "data/chatgpt_memories.json" "ChatGPT conversations" "29MB"

# FAISS vector index
transfer_file "data/faiss_chatgpt_index.index" "FAISS vector embeddings" "139MB"

# FAISS metadata
transfer_file "data/faiss_chatgpt_index.pkl" "FAISS metadata" "303MB"

# Transfer other essential data files
echo -e "\n${YELLOW}📋 Syncing configuration files...${NC}"
scp -i $EC2_KEY data/import_progress.json "$EC2_HOST:$PROJECT_DIR/data/" 2>/dev/null

# Verify the transfer
echo -e "\n${YELLOW}🔍 Verifying transfer...${NC}"
ssh -i $EC2_KEY $EC2_HOST << 'ENDSSH'
cd ~/ai-memory-layer/data
echo "Files on EC2:"
ls -lh chatgpt_memories.json faiss_chatgpt_index.* | awk '{print "  ✓ " $9 ": " $5}'

# Count memories
if [ -f chatgpt_memories.json ]; then
    count=$(grep -o '"content":' chatgpt_memories.json | wc -l)
    echo ""
    echo "  📊 Total memories: $count"
fi
ENDSSH

echo -e "\n${GREEN}✅ Data sync complete!${NC}"
echo "========================================"
echo -e "${GREEN}🎉 Success! Your ChatGPT memory database is now on EC2${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. SSH to EC2: ssh -i $EC2_KEY $EC2_HOST"
echo "2. cd $PROJECT_DIR"
echo "3. Test the memory search: python example.py"
echo "4. Or start the API: python run_api.py"
echo ""
echo -e "${BLUE}Your 23,234 conversations are ready for AI-powered search!${NC}"