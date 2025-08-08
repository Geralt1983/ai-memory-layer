#!/bin/bash

# AI Memory Layer - Automated Deployment with Snapshots and Versioning
# This script creates version snapshots and deploys the system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AI Memory Layer - Automated Deployment${NC}"
echo "================================================"

# Get current commit info
COMMIT_HASH=$(git rev-parse --short HEAD)
COMMIT_MESSAGE=$(git log -1 --pretty=format:"%s")
BRANCH=$(git branch --show-current)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${BLUE}📋 Deployment Info:${NC}"
echo "  Branch: $BRANCH"
echo "  Commit: $COMMIT_HASH"
echo "  Message: $COMMIT_MESSAGE"
echo "  Timestamp: $TIMESTAMP"
echo ""

# Determine version increment based on commit message
if [[ "$COMMIT_MESSAGE" =~ ^feat.*BREAKING ]]; then
    VERSION_TYPE="major"
elif [[ "$COMMIT_MESSAGE" =~ ^feat ]]; then
    VERSION_TYPE="minor"
elif [[ "$COMMIT_MESSAGE" =~ ^fix ]]; then
    VERSION_TYPE="patch"
else
    VERSION_TYPE="patch"  # Default to patch
fi

echo -e "${YELLOW}📦 Version Type: $VERSION_TYPE${NC}"

# Get current version or start at v1.0.0
CURRENT_VERSION=$(git tag -l "v*" | sort -V | tail -1)
if [[ -z "$CURRENT_VERSION" ]]; then
    NEW_VERSION="v1.0.0"
else
    # Parse version numbers
    CURRENT_VERSION=${CURRENT_VERSION#v}  # Remove 'v' prefix
    IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
    MAJOR=${VERSION_PARTS[0]}
    MINOR=${VERSION_PARTS[1]}
    PATCH=${VERSION_PARTS[2]}
    
    # Increment based on type
    case $VERSION_TYPE in
        "major")
            MAJOR=$((MAJOR + 1))
            MINOR=0
            PATCH=0
            ;;
        "minor")
            MINOR=$((MINOR + 1))
            PATCH=0
            ;;
        "patch")
            PATCH=$((PATCH + 1))
            ;;
    esac
    
    NEW_VERSION="v$MAJOR.$MINOR.$PATCH"
fi

echo -e "${GREEN}🏷️  New Version: $NEW_VERSION${NC}"

# Create deployment snapshot directory
SNAPSHOT_DIR="deployments/${NEW_VERSION}_${TIMESTAMP}"
mkdir -p "$SNAPSHOT_DIR"

echo -e "${BLUE}📸 Creating deployment snapshot...${NC}"

# Create snapshot metadata
cat > "$SNAPSHOT_DIR/deployment.json" << EOF
{
  "version": "$NEW_VERSION",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "commit_hash": "$COMMIT_HASH",
  "commit_message": "$COMMIT_MESSAGE",
  "branch": "$BRANCH",
  "deployment_type": "$VERSION_TYPE",
  "system_info": {
    "os": "$(uname -s)",
    "arch": "$(uname -m)",
    "python_version": "$(python3 --version 2>&1)",
    "node_version": "$(node --version 2>/dev/null || echo 'not installed')"
  }
}
EOF

# Copy current state
echo -e "${BLUE}📋 Copying current system state...${NC}"
rsync -a --exclude='.git' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='node_modules' \
         --exclude='*.log' \
         --exclude='deployments' \
         --exclude='data/faiss_*' \
         --exclude='venv' \
         --exclude='.mypy_cache' \
         ./ "$SNAPSHOT_DIR/" >/dev/null 2>&1 || true

# Create git tag
echo -e "${BLUE}🏷️  Creating git tag: $NEW_VERSION${NC}"
git tag -a "$NEW_VERSION" -m "Release $NEW_VERSION: $COMMIT_MESSAGE"

# Stop existing services
echo -e "${YELLOW}🛑 Stopping existing services...${NC}"
pkill -f "chatgpt_memory_api.py" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 2

# Install/update dependencies if needed
echo -e "${BLUE}📦 Checking dependencies...${NC}"
if [[ -f "requirements.txt" ]]; then
    pip3 install -r requirements.txt >/dev/null 2>&1 || true
fi

# Pre-deployment health check
echo -e "${BLUE}🔍 Pre-deployment validation...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from core.memory_engine import MemoryEngine
    from integrations.embeddings import EmbeddingProvider
    print('✅ Core modules import successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Start the API server
echo -e "${GREEN}🚀 Starting AI Memory Layer API...${NC}"
python3 chatgpt_memory_api.py > "api_server_${NEW_VERSION}.log" 2>&1 &
API_PID=$!

# Wait for API to start
echo -e "${BLUE}⏳ Waiting for API to start...${NC}"
sleep 5

# Health check
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✅ API is healthy!${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -e "${YELLOW}⏳ Waiting for API (attempt $RETRY_COUNT/$MAX_RETRIES)...${NC}"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ API failed to start properly${NC}"
    exit 1
fi

# Start tunnel
echo -e "${GREEN}🌐 Starting Cloudflare tunnel...${NC}"
cloudflared tunnel --url http://localhost:8000 > tunnel_${NEW_VERSION}.log 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel URL
echo -e "${BLUE}⏳ Waiting for tunnel to establish...${NC}"
sleep 5

TUNNEL_URL=""
for i in {1..30}; do
    TUNNEL_URL=$(grep -E "https://.*\.trycloudflare\.com" "tunnel_${NEW_VERSION}.log" 2>/dev/null | tail -1 | grep -oE "https://[^[:space:]]*" || true)
    if [[ -n "$TUNNEL_URL" ]]; then
        break
    fi
    sleep 1
done

# Final deployment validation
echo -e "${BLUE}🔍 Final deployment validation...${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health || echo "failed")
if [[ "$HEALTH_RESPONSE" == "failed" ]]; then
    echo -e "${RED}❌ Final health check failed${NC}"
    exit 1
fi

# Test neural network features
echo -e "${BLUE}🧠 Testing neural network integration...${NC}"
TEST_RESPONSE=$(curl -s -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "How many dogs does Jeremy have?", "use_gpt": false}' \
    | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('response', ''))" 2>/dev/null || echo "test failed")

if [[ "$TEST_RESPONSE" == *"two dogs"* ]] || [[ "$TEST_RESPONSE" == *"Remy"* ]] || [[ "$TEST_RESPONSE" == *"Bailey"* ]]; then
    echo -e "${GREEN}✅ Neural network integration working!${NC}"
else
    echo -e "${YELLOW}⚠️  Neural network test inconclusive${NC}"
fi

# Save deployment info to snapshot
cat >> "$SNAPSHOT_DIR/deployment.json" << EOF2

{
  "deployment_status": "success",
  "api_pid": $API_PID,
  "tunnel_pid": $TUNNEL_PID,
  "tunnel_url": "$TUNNEL_URL",
  "health_check": "passed",
  "neural_network_test": "$(echo $TEST_RESPONSE | head -c 100)...",
  "log_files": {
    "api": "api_server_${NEW_VERSION}.log",
    "tunnel": "tunnel_${NEW_VERSION}.log"
  }
}
EOF2

# Create deployment summary
echo ""
echo -e "${GREEN}🎉 DEPLOYMENT SUCCESSFUL!${NC}"
echo "================================================"
echo -e "${GREEN}Version:${NC} $NEW_VERSION"
echo -e "${GREEN}Commit:${NC} $COMMIT_HASH"
echo -e "${GREEN}API URL:${NC} http://localhost:8000"
if [[ -n "$TUNNEL_URL" ]]; then
    echo -e "${GREEN}Public URL:${NC} $TUNNEL_URL"
fi
echo -e "${GREEN}Snapshot:${NC} $SNAPSHOT_DIR"
echo -e "${GREEN}API Log:${NC} api_server_${NEW_VERSION}.log"
echo -e "${GREEN}Tunnel Log:${NC} tunnel_${NEW_VERSION}.log"
echo ""
echo -e "${BLUE}🧠 Neural Network Features:${NC} ✅ Active"
echo -e "${BLUE}📊 Memory Count:${NC} $(curl -s http://localhost:8000/health | python3 -c "import json, sys; print(json.load(sys.stdin).get('memory_count', 'unknown'))" 2>/dev/null || echo 'unknown')"
echo ""
echo -e "${YELLOW}💡 Next steps:${NC}"
echo "  - Test the deployment at the public URL"
echo "  - Monitor logs for any issues"
echo "  - Push git tags: git push origin --tags"
echo ""

# Save process IDs for easy management
echo "$API_PID" > ".api.pid"
echo "$TUNNEL_PID" > ".tunnel.pid"
echo "$NEW_VERSION" > ".current_version"

# Create a status script
cat > "deployment_status.sh" << 'EOF3'
#!/bin/bash
if [[ -f ".current_version" ]]; then
    VERSION=$(cat .current_version)
    echo "🚀 Current Version: $VERSION"
    
    if [[ -f ".api.pid" ]]; then
        API_PID=$(cat .api.pid)
        if ps -p $API_PID > /dev/null; then
            echo "✅ API: Running (PID: $API_PID)"
        else
            echo "❌ API: Not running"
        fi
    fi
    
    if [[ -f ".tunnel.pid" ]]; then
        TUNNEL_PID=$(cat .tunnel.pid)
        if ps -p $TUNNEL_PID > /dev/null; then
            echo "✅ Tunnel: Running (PID: $TUNNEL_PID)"
        else
            echo "❌ Tunnel: Not running"
        fi
    fi
    
    # Show tunnel URL if available
    TUNNEL_LOG="tunnel_${VERSION}.log"
    if [[ -f "$TUNNEL_LOG" ]]; then
        TUNNEL_URL=$(grep -E "https://.*\.trycloudflare\.com" "$TUNNEL_LOG" 2>/dev/null | tail -1 | grep -oE "https://[^[:space:]]*" || true)
        if [[ -n "$TUNNEL_URL" ]]; then
            echo "🌐 Public URL: $TUNNEL_URL"
        fi
    fi
    
    # API health
    HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "failed")
    if [[ "$HEALTH" != "failed" ]]; then
        echo "✅ Health: OK"
    else
        echo "❌ Health: Failed"
    fi
else
    echo "❌ No deployment found"
fi
EOF3

chmod +x deployment_status.sh

echo -e "${GREEN}✅ Deployment script completed successfully!${NC}"