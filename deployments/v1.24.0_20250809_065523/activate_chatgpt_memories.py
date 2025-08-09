#!/usr/bin/env python3
"""
Activate Full ChatGPT Memory System
===================================

This script replaces the current API with the full 23,710 ChatGPT memory system.
Run this on the EC2 server to activate the complete solution.
"""

import os
import sys
import subprocess
import time

def main():
    print("🚀 Activating Full ChatGPT Memory System")
    print("=" * 50)
    
    # Change to project directory
    os.chdir('/home/ubuntu/ai-memory-layer')
    
    # Check if we have all required files
    required_files = [
        'data/chatgpt_memories.json',
        'data/faiss_chatgpt_index.index', 
        'data/faiss_chatgpt_index.pkl',
        'run_chatgpt_system.py',
        'optimized_memory_loader.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"✅ {file_path} ({size_mb:.1f}MB)")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("\n🛑 Stopping current API server...")
    
    # Stop any running Python processes
    try:
        subprocess.run(['pkill', '-f', 'python.*api'], check=False)
        subprocess.run(['pkill', '-f', 'uvicorn'], check=False)
        time.sleep(3)
    except:
        pass
    
    print("🚀 Starting ChatGPT Memory API...")
    
    # Activate virtual environment and start our system
    try:
        env = os.environ.copy()
        env['PATH'] = '/home/ubuntu/ai-memory-layer/venv/bin:' + env['PATH']
        
        # Start the full ChatGPT system
        result = subprocess.run([
            '/home/ubuntu/ai-memory-layer/venv/bin/python',
            'run_chatgpt_system.py'
        ], env=env, check=True)
        
        print("✅ ChatGPT Memory API started successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start ChatGPT API: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)