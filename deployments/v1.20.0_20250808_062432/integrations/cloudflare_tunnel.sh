#!/bin/bash
"""
Cloudflare Tunnel Setup for Stable AI Memory Layer Access
Replaces ngrok with permanent, reliable URL
"""

# Configuration
TUNNEL_NAME="ai-memory-layer"
LOCAL_PORT=8000
SERVICE_URL="http://localhost:${LOCAL_PORT}"

echo "🌐 Setting up Cloudflare Tunnel for AI Memory Layer"
echo "📍 Local service: ${SERVICE_URL}"
echo "🔗 Tunnel name: ${TUNNEL_NAME}"

# Check if tunnel already exists
existing_tunnel=$(cloudflared tunnel list 2>/dev/null | grep ${TUNNEL_NAME} | awk '{print $1}')

if [ -n "$existing_tunnel" ]; then
    echo "✅ Found existing tunnel: ${existing_tunnel}"
    TUNNEL_ID="$existing_tunnel"
else
    echo "🆕 Creating new tunnel: ${TUNNEL_NAME}"
    
    # Create new tunnel
    TUNNEL_ID=$(cloudflared tunnel create ${TUNNEL_NAME} 2>&1 | grep -o '[a-z0-9-]\{36\}')
    
    if [ -n "$TUNNEL_ID" ]; then
        echo "✅ Created tunnel with ID: ${TUNNEL_ID}"
    else
        echo "❌ Failed to create tunnel"
        exit 1
    fi
fi

# Create tunnel config file
CONFIG_FILE="$HOME/.cloudflared/config.yml"
echo "📝 Creating tunnel configuration..."

mkdir -p "$HOME/.cloudflared"

cat > "$CONFIG_FILE" << EOF
tunnel: ${TUNNEL_ID}
credentials-file: $HOME/.cloudflared/${TUNNEL_ID}.json

ingress:
  - hostname: ${TUNNEL_NAME}.trycloudflare.com
    service: ${SERVICE_URL}
    originRequest:
      httpHostHeader: localhost:${LOCAL_PORT}
  - service: http_status:404

EOF

echo "✅ Tunnel configuration saved to: ${CONFIG_FILE}"

# Start tunnel
echo "🚀 Starting Cloudflare tunnel..."
echo "🔗 Your AI Memory Layer will be available at: https://${TUNNEL_NAME}.trycloudflare.com"
echo ""
echo "Press Ctrl+C to stop the tunnel"
echo ""

# Run tunnel
cloudflared tunnel run ${TUNNEL_NAME}