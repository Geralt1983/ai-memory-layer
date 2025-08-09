#!/usr/bin/env python3
"""
Check Import Status
Monitor the progress of the ChatGPT import without interrupting it
"""

import json
import os
from datetime import datetime
from pathlib import Path

def check_status():
    """Check the current import status"""
    progress_file = "./data/import_progress.json"
    memories_file = "./data/chatgpt_memories.json"
    
    print("🔍 ChatGPT Import Status Check")
    print("=" * 50)
    
    # Check progress file
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        processed = progress.get("processed_count", 0)
        total = progress.get("total_count", 0)
        errors = progress.get("errors", 0)
        completion = progress.get("completion_percentage", 0)
        timestamp = progress.get("timestamp", "Unknown")
        
        print(f"📊 Progress: {processed:,} / {total:,} messages ({completion:.1f}%)")
        print(f"❌ Errors: {errors}")
        print(f"🕐 Last Update: {timestamp}")
        
        if completion > 0:
            # Estimate completion time
            time_per_message = 0.35  # Rough estimate based on logs
            remaining = total - processed
            eta_seconds = remaining * time_per_message
            eta_hours = eta_seconds / 3600
            print(f"⏱️  Estimated Time Remaining: {eta_hours:.1f} hours")
        
    else:
        print("📂 No progress file found - import may not have started")
    
    # Check memory file
    if os.path.exists(memories_file):
        try:
            with open(memories_file, 'r') as f:
                memories = json.load(f)
            print(f"💾 Memories Stored: {len(memories):,}")
            
            # Show sample of recent memories
            if memories:
                recent = memories[-3:]
                print(f"\n📝 Recent Imports:")
                for i, mem in enumerate(recent):
                    role = mem.get('role', 'unknown')
                    title = mem.get('title', 'No title')[:50]
                    content = mem.get('content', '')[:80]
                    importance = mem.get('importance', 0)
                    print(f"  {i+1}. [{role}] {title}... (importance: {importance:.2f})")
                    print(f"     Content: {content}...")
                    print()
        except:
            print("💾 Memory file exists but couldn't read it")
    else:
        print("💾 No memory file found yet")
    
    # Check vector store
    vector_dir = "./data/faiss_chatgpt_index"
    if os.path.exists(vector_dir):
        files = list(Path(vector_dir).glob("*"))
        print(f"🗂️  Vector Store Files: {len(files)} files")
    else:
        print("🗂️  Vector store not created yet")
    
    print("\n" + "=" * 50)
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        completion = progress.get("completion_percentage", 0)
        
        if completion >= 100:
            print("✅ Import Complete!")
            print("🚀 Your ChatGPT conversations are now searchable with AI embeddings!")
        elif completion > 0:
            print("⏳ Import in progress...")
            print("💡 You can safely close this and check back later")
            print("💡 The import will resume automatically if interrupted")
        else:
            print("🔄 Import may be starting...")
    
    return True

if __name__ == "__main__":
    check_status()