#!/usr/bin/env python3
"""
Local ChatGPT Memory System Runner
==================================

Run this locally to start the ChatGPT memory system with web interface.
No server management needed - just run and go!
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🚀 Starting Local ChatGPT Memory System")
    print("=" * 50)
    
    # Check if we have the data files
    data_files = [
        "./data/chatgpt_memories.json",
        "./data/faiss_chatgpt_index.index",
        "./data/faiss_chatgpt_index.pkl"
    ]
    
    missing = []
    for file_path in data_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"✅ {file_path} ({size_mb:.1f}MB)")
        else:
            missing.append(file_path)
    
    if missing:
        print(f"❌ Missing data files: {missing}")
        print("💡 The ChatGPT data wasn't copied locally")
        print("🔧 Let's create a demo with your existing data instead...")
        
        # Use the regular API with existing data
        try:
            from run_api import main as run_regular_api
            run_regular_api()
        except:
            print("❌ No API available. Please run: python run_api.py")
        return
    
    # We have the ChatGPT data! Load it
    print("\n📚 Loading ChatGPT Memory System...")
    try:
        from run_chatgpt_system import main as run_chatgpt
        
        # Open browser automatically
        import threading
        def open_browser():
            time.sleep(3)  # Wait for server to start
            webbrowser.open("http://localhost:8000")
            print("\n🌐 Opened http://localhost:8000 in your browser")
            print("🎉 ChatGPT Memory System with 23,710 conversations is ready!")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start the ChatGPT system
        run_chatgpt()
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Falling back to regular API...")
        try:
            from run_api import main as run_regular_api
            run_regular_api()
        except:
            print("❌ Could not start any API. Check your setup.")

if __name__ == "__main__":
    main()