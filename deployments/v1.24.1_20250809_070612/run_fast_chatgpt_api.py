#!/usr/bin/env python3
"""
Ultra-Fast ChatGPT API Startup
==============================

Instantly loads 23,710 ChatGPT memories using pre-computed FAISS embeddings.
No re-embedding, no delays - just fast memory access.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    print("⚡ Ultra-Fast ChatGPT Memory API")
    print("=" * 40)
    
    # Load environment
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv()
            print("✓ Environment loaded")
    except ImportError:
        print("⚠️ python-dotenv not available")
    
    # Verify API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required")
        sys.exit(1)
    
    # Patch the API with ChatGPT data BEFORE importing the app
    print("🧠 Patching API with ChatGPT data...")
    
    try:
        from fast_chatgpt_loader import patch_api_startup
        success = patch_api_startup()
        
        if not success:
            print("❌ Failed to patch API with ChatGPT data")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Patcher error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Now start the API server
    try:
        import uvicorn
        from api.main import app
        
        print("\n🌐 Starting FastAPI server...")
        print("🔍 Memory Search: http://localhost:8000/memories/search") 
        print("💬 Chat Interface: http://localhost:8000/")
        print("📊 Stats: http://localhost:8000/stats")
        print("📚 23,710 ChatGPT memories ready!")
        print("\nPress Ctrl+C to stop")
        print("=" * 40)
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()