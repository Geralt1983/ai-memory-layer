#!/bin/bash
# AI Memory Layer Deployment Script with Auto-Versioning

echo "🚀 Deploying AI Memory Layer to EC2..."

# Variables
EC2_HOST="ubuntu@18.224.179.36"
EC2_KEY="~/.ssh/AI-memory.pem"
PROJECT_DIR="~/ai-memory-layer"

# Auto-increment version
CURRENT_VERSION=$(cat VERSION)
IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

# Determine version bump based on commit message
if [[ "$1" == *"BREAKING:"* ]] || [[ "$1" == *"breaking:"* ]]; then
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
    echo "📈 Major version bump (breaking changes)"
elif [[ "$1" == *"feat:"* ]] || [[ "$1" == *"feature:"* ]]; then
    MINOR=$((MINOR + 1))
    PATCH=0
    echo "📈 Minor version bump (new feature)"
else
    PATCH=$((PATCH + 1))
    echo "📈 Patch version bump (fixes/improvements)"
fi

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "📌 Version: $CURRENT_VERSION → $NEW_VERSION"

# Update VERSION file
echo "$NEW_VERSION" > VERSION

# Update README.md version
sed -i '' "s/\*\*v$CURRENT_VERSION\*\*/\*\*v$NEW_VERSION\*\*/g" README.md
sed -i '' "s/Version\*\*: v$CURRENT_VERSION/Version\*\*: v$NEW_VERSION/g" README.md

# Add changelog entry
DATE=$(date +%Y-%m-%d)
CHANGELOG_ENTRY="## [$NEW_VERSION] - $DATE\n\n### Changed\n- $1\n"

# Insert after the first ## [version] line
awk -v entry="$CHANGELOG_ENTRY" '/^## \[.*\]/ && !inserted {print entry; inserted=1} {print}' CHANGELOG.md > CHANGELOG.tmp && mv CHANGELOG.tmp CHANGELOG.md

# Git operations
echo "📦 Committing and pushing to git..."
git add VERSION README.md CHANGELOG.md
git add -A
git commit -m "$1 (v$NEW_VERSION)"
git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"
git push origin main --tags

# Sync ChatGPT Memory Data
echo "🧠 Syncing ChatGPT memory database..."
if [ -f "data/chatgpt_memories.json" ]; then
    MEMORY_SIZE=$(ls -lh data/chatgpt_memories.json | awk '{print $5}')
    echo "  📊 Transferring ChatGPT memories ($MEMORY_SIZE)..."
    rsync -avz --progress -e "ssh -i $EC2_KEY" \
        data/chatgpt_memories.json \
        $EC2_HOST:$PROJECT_DIR/data/
    
    echo "  📦 Transferring FAISS indexes..."
    rsync -avz --progress -e "ssh -i $EC2_KEY" \
        data/faiss_chatgpt_index.index \
        data/faiss_chatgpt_index.pkl \
        $EC2_HOST:$PROJECT_DIR/data/
    
    # Verify transfer
    ssh -i $EC2_KEY $EC2_HOST "cd $PROJECT_DIR && python3 -c \"import json; data=json.load(open('data/chatgpt_memories.json')); print('✅ Verified:', len(data), 'memories on EC2')\""
else
    echo "  ⚠️  No ChatGPT memory data found locally"
fi

# Deploy to EC2
echo "🔄 Deploying code to EC2..."
ssh -i $EC2_KEY $EC2_HOST << ENDSSH
cd ~/ai-memory-layer
git pull origin main
source venv/bin/activate
pip install -r requirements.txt

# Update the web interface with version
# First update the version in the source file
sed -i "s/v[0-9]*\.[0-9]*\.[0-9]*/v$NEW_VERSION/g" ~/ai-memory-layer/web_interface_enhanced.html

# Then copy to nginx directory
if [ -f /var/www/html/index.html ]; then
    sudo cp ~/ai-memory-layer/web_interface_enhanced.html /var/www/html/index.html
    echo "Updated /var/www/html/index.html with version v$NEW_VERSION"
else
    echo "Web interface served from project directory"
fi

# Deploy AI Code Assistant
echo "🤖 Deploying AI Code Assistant..."
cd ~/ai-memory-layer

# Sync AI assistant files
rsync -avz --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    ./ai_code_assistant/ \
    ./ai_code_assistant/

# Install AI assistant dependencies
cd ai_code_assistant
pip install -r requirements.txt

# Index commits for AI assistant (if not already done)
if [ ! -f "data/metadata.db" ]; then
    echo "🔍 Initial indexing of commits for AI assistant..."
    python main.py --index-commits --git-dir=../
fi

# Update AI assistant service
sudo systemctl daemon-reload
sudo systemctl restart ai-code-assistant.service || echo "AI assistant service not configured yet"

# Restart main service
sudo systemctl restart ai-memory.service
echo "✅ Deployment complete! Version $NEW_VERSION is live."
ENDSSH

echo "🎉 Successfully deployed v$NEW_VERSION to http://18.224.179.36"
echo ""
echo "📝 Commit conventions for auto-versioning:"
echo "  • 'fix: ...' or default → Patch (0.0.X)"
echo "  • 'feat: ...' → Minor (0.X.0)"
echo "  • 'BREAKING: ...' → Major (X.0.0)"