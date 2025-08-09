#!/bin/bash
# Simple script to start the ChatGPT Memory System

cd ~/ai-memory-layer
source venv/bin/activate

# Kill any existing API processes
echo "🛑 Stopping existing processes..."
pkill -f "python.*api" || true
pkill -f fixed_direct || true
sleep 2

# Start the ChatGPT Memory API
echo "🚀 Starting ChatGPT Memory API with 23,710 memories..."
python fix_api_complete.py