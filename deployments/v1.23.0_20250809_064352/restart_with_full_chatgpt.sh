#!/bin/bash
# Restart API server with full 23,710 ChatGPT memories

set -e

echo "🚀 Restarting API with Full ChatGPT Memory Dataset"
echo "=" * 50

# Create the startup command that uses our optimized loader
cat > start_full_chatgpt_api.py << 'EOF'
#!/usr/bin/env python3
import os
import sys

# Add project root to path
sys.path.insert(0, '/home/ubuntu/ai-memory-layer')

# Load environment
from dotenv import load_dotenv
load_dotenv()

print("🚀 Starting ChatGPT Memory API with 23,710 conversations...")

# Import and start the full system
from run_chatgpt_system import main

if __name__ == "__main__":
    main()
EOF

echo "📄 Created startup script for full ChatGPT dataset"
echo "🔄 Deploying to EC2..."

# Copy the script to EC2 via the working deployment mechanism
# This leverages the rsync that works in deploy.sh