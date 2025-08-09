#!/usr/bin/env python3
"""
Lightweight ChatGPT API Runner
=============================

A memory-efficient version that loads ChatGPT memories without heavy optimizations.
Designed for resource-constrained environments.
"""

import os
import sys
import time
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
try:
    from dotenv import load_dotenv
    if os.path.exists(".env"):
        load_dotenv()
        print("✅ Environment loaded from .env")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment")


def cleanup_processes():
    """Clean up any existing Python processes"""
    import subprocess
    try:
        subprocess.run(["pkill", "-f", "python"], check=False)
        subprocess.run(["pkill", "-f", "uvicorn"], check=False)
        time.sleep(2)
        print("🧹 Cleaned up existing processes")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


def start_lightweight_api():
    """Start lightweight API with existing optimized loader"""
    print("🚀 Lightweight ChatGPT API Startup")
    print("=" * 40)
    
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required")
        sys.exit(1)
    
    # Check data files
    data_files = [
        "./data/chatgpt_memories.json",
        "./data/faiss_chatgpt_index.index", 
        "./data/faiss_chatgpt_index.pkl"
    ]
    
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            sys.exit(1)
        else:
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"✅ {file_path} ({size_mb:.1f}MB)")
    
    try:
        # Use existing optimized loader (lighter than ultra-optimized)
        from optimized_chatgpt_loader import create_optimized_api
        
        print("📚 Loading ChatGPT memories (optimized approach)...")
        start_time = time.time()
        
        memory_engine = create_optimized_api(
            faiss_index_path="./data/faiss_chatgpt_index",
            memory_json_path="./data/chatgpt_memories.json", 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        load_time = time.time() - start_time
        print(f"✅ Memory engine ready in {load_time:.2f}s")
        print(f"📊 Loaded {len(memory_engine.memories)} memories")
        
        # Start API with the loaded engine
        from api.main import app
        import api.main as api_main
        
        # Replace global memory engine
        api_main.memory_engine = memory_engine
        print("🔄 Integrated with API")
        
        # Start server
        import uvicorn
        print("\n🌐 Starting API server...")
        print("📍 Health: http://localhost:8000/health")
        print("🔍 Search: http://localhost:8000/memories/search")
        print("💬 Chat: http://localhost:8000/chat")
        print("\nPress Ctrl+C to stop")
        print("=" * 40)
        
        uvicorn.run(
            app,
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            access_log=False
        )
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Shutting down...")
    sys.exit(0)


def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Clean up and start
    cleanup_processes()
    start_lightweight_api()


if __name__ == "__main__":
    main()