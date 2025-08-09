#!/usr/bin/env python3
"""
ChatGPT Memory API Runner - Full API with ChatGPT Data
=====================================================

Modified version of run_api.py that uses the 23,710 ChatGPT memories
instead of the default test memories.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment")
        return False

    return True


def load_chatgpt_memory_engine():
    """Create memory engine with ChatGPT data"""
    print("🚀 Loading ChatGPT Memory Engine...")
    
    try:
        # Load ChatGPT data
        start_time = time.time()
        
        import faiss
        faiss_path = "./data/faiss_chatgpt_index"
        index = faiss.read_index(f"{faiss_path}.index")
        
        with open("./data/chatgpt_memories.json", 'r') as f:
            memory_data = json.load(f)
        
        load_time = time.time() - start_time
        print(f"✅ Loaded {index.ntotal} FAISS vectors and {len(memory_data)} memories in {load_time:.2f}s")
        
        # Create optimized memory engine using existing loader
        from optimized_chatgpt_loader import create_optimized_api
        
        memory_engine = create_optimized_api(
            faiss_index_path="./data/faiss_chatgpt_index",
            memory_json_path="./data/chatgpt_memories.json",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print(f"🎉 ChatGPT Memory Engine ready with {len(memory_engine.memories)} memories")
        return memory_engine
        
    except Exception as e:
        print(f"❌ Error loading ChatGPT data: {e}")
        # Fallback to regular API
        return None


def main():
    print("AI Memory Layer API Server with ChatGPT Data")
    print("=" * 50)

    # Load environment variables from .env if available
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv()
            print("✓ Loaded environment variables from .env")
        else:
            print("! No .env file found, using system environment variables")
    except ImportError:
        print("! python-dotenv not installed, using system environment variables")

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Load ChatGPT memory engine
    chatgpt_engine = load_chatgpt_memory_engine()
    
    # Import and modify the API
    try:
        import uvicorn
        from api.main import app
        import api.main as api_main

        # Replace the API's memory engine with ChatGPT data if loaded
        if chatgpt_engine:
            print("🔄 Replacing API memory engine with ChatGPT data...")
            
            # Monkey patch the startup to use our engine
            original_startup = api_main.startup_event
            
            async def chatgpt_startup():
                print("📚 Using pre-loaded ChatGPT memory engine...")
                api_main.memory_engine = chatgpt_engine
                
                # Initialize other components normally
                from integrations.direct_openai import DirectOpenAIChat
                from core.memory_manager import create_default_memory_manager
                
                api_main.direct_openai_chat = DirectOpenAIChat(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    memory_engine=chatgpt_engine,
                    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                    system_prompt_path=os.getenv("SYSTEM_PROMPT_PATH", "./prompts/system_prompt_4o.txt"),
                )
                
                api_main.memory_manager = create_default_memory_manager(chatgpt_engine)
                print(f"✅ API initialized with {len(chatgpt_engine.memories)} ChatGPT memories")
            
            # Replace the startup function
            app.router.lifespan_context = lambda app: chatgpt_startup()

        print("\n🚀 Starting API server...")
        print("📝 API Documentation: http://localhost:8000/docs")
        print("❤️  Health Check: http://localhost:8000/health")
        print("🔍 OpenAPI Schema: http://localhost:8000/openapi.json")
        print("💬 Chat Endpoint: http://localhost:8000/chat")
        if chatgpt_engine:
            print(f"📊 ChatGPT Memories: {len(chatgpt_engine.memories):,}")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()