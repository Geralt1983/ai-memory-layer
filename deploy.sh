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

# Deploy to EC2
echo "🔄 Deploying to EC2..."
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

# Restart service
sudo systemctl restart ai-memory.service
echo "✅ Deployment complete! Version $NEW_VERSION is live."
ENDSSH

echo "🎉 Successfully deployed v$NEW_VERSION to http://18.224.179.36"
echo ""
echo "📝 Commit conventions for auto-versioning:"
echo "  • 'fix: ...' or default → Patch (0.0.X)"
echo "  • 'feat: ...' → Minor (0.X.0)"
echo "  • 'BREAKING: ...' → Major (X.0.0)"