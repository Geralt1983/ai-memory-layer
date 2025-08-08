#!/bin/bash
# Controlled startup script for ultra-optimized ChatGPT API
# This ensures only one process runs and monitors resources

echo "🧹 Cleaning up any existing processes..."
pkill -f python
pkill -f uvicorn
sleep 2

echo "📊 Checking system resources before startup..."
free -h
echo "CPU usage:"
top -b -n 1 | grep "Cpu(s)"

echo "🔍 Verifying data files..."
ls -lh ~/ai-memory-layer/data/chatgpt_memories.json
ls -lh ~/ai-memory-layer/data/faiss_chatgpt_index.*

echo "🚀 Starting ultra-optimized API with resource monitoring..."
cd ~/ai-memory-layer
source venv/bin/activate

# Start with nohup and redirect output
nohup python run_ultra_optimized_chatgpt_api.py > ultra_startup.log 2>&1 &
PYTHON_PID=$!

echo "🔄 Started Python process PID: $PYTHON_PID"
echo "📝 Logs are in: ~/ai-memory-layer/ultra_startup.log"

# Monitor for 30 seconds
for i in {1..6}; do
    sleep 5
    echo "⏱️  ${i}0s - Checking process status..."
    if ps -p $PYTHON_PID > /dev/null; then
        echo "✅ Process $PYTHON_PID still running"
        echo "💾 Memory usage:"
        ps -p $PYTHON_PID -o pid,ppid,%mem,%cpu,cmd
    else
        echo "❌ Process $PYTHON_PID has stopped"
        echo "📜 Last few log lines:"
        tail -10 ultra_startup.log
        exit 1
    fi
done

echo "🎉 Startup monitoring complete. Process should be ready."
echo "🌐 Test with: curl http://localhost:8000/health"