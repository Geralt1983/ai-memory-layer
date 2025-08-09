#!/usr/bin/env python3
"""
Production API with Optimized ChatGPT Memory Loading
===================================================

Fast-loading API that serves 23,710 ChatGPT memories without re-embedding.
Uses pre-computed FAISS vectors paired with memory content.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from optimized_chatgpt_loader import create_optimized_api


def main():
    print("🚀 Starting Optimized ChatGPT Memory API")
    print("=" * 50)
    
    # Load environment
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv()
            print("✓ Loaded environment variables")
    except ImportError:
        print("⚠️ python-dotenv not available")
    
    # Get configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY required")
        sys.exit(1)
    
    faiss_index_path = "./data/faiss_chatgpt_index"
    memory_json_path = "./data/chatgpt_memories.json"
    
    # Create optimized memory engine
    print("🧠 Initializing optimized memory engine...")
    try:
        memory_engine = create_optimized_api(
            faiss_index_path=faiss_index_path,
            memory_json_path=memory_json_path,
            openai_api_key=openai_api_key
        )
        print(f"✅ Memory engine ready with {len(memory_engine.memories)} memories")
        
    except Exception as e:
        print(f"❌ Failed to initialize memory engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Start API server
    try:
        import uvicorn
        from api.main import app
        
        # Replace the global memory engine in the API
        import api.main as api_main
        api_main.memory_engine = memory_engine
        
        print("\n🌐 Starting FastAPI server...")
        print("📚 Memory Search: http://localhost:8000/memories/search")
        print("💬 Chat Interface: http://localhost:8000/")
        print("📊 Stats: http://localhost:8000/stats")
        print("\nPress Ctrl+C to stop")
        print("=" * 50)
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()