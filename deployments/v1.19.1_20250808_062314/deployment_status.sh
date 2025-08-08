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
