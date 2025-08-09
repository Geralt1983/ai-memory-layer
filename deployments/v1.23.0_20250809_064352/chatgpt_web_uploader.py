#!/usr/bin/env python3
"""
ChatGPT Web Interface Uploader
===============================

Alternative approach: Instead of API threads, this creates files optimized
for manual upload to ChatGPT web interface conversations.

Features:
- Creates ChatGPT-optimized summaries
- Formats for easy copy-paste or drag-drop
- Maintains conversation context
- Works with any ChatGPT conversation

Usage:
    python chatgpt_web_uploader.py --process-latest
    python chatgpt_web_uploader.py --process-all
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

SYNC_DIR = Path("./chatgpt-sync")
OUTPUT_DIR = Path("./chatgpt-ready")

class ChatGPTWebUploader:
    """Prepare sync files for ChatGPT web interface"""
    
    def __init__(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
    
    def process_sync_file(self, md_file: Path) -> Path:
        """Convert webhook sync file to ChatGPT-optimized format"""
        
        content = md_file.read_text()
        
        # Enhanced format for ChatGPT web interface
        enhanced_content = f"""📎 **AI Memory Layer - Automated Code Update**

{content}

---

🤖 **Instructions for AI Assistant:**
- This is an automated update from my AI Memory Layer GitHub repository
- Please review the code changes and provide feedback
- Identify any potential issues, improvements, or questions
- Maintain context of this project's evolution
- Ask if you need clarification on any changes

**Ready for your analysis and feedback!** ✨"""

        # Create output file
        output_file = OUTPUT_DIR / f"chatgpt_{md_file.name}"
        output_file.write_text(enhanced_content)
        
        return output_file
    
    def process_latest(self) -> Optional[Path]:
        """Process the latest sync file"""
        if not SYNC_DIR.exists():
            print(f"❌ Sync directory not found: {SYNC_DIR}")
            return None
        
        md_files = list(SYNC_DIR.glob("*.md"))
        if not md_files:
            print("❌ No .md files found")
            return None
        
        latest_file = max(md_files, key=lambda f: f.stat().st_mtime)
        output_file = self.process_sync_file(latest_file)
        
        print(f"✅ Processed: {latest_file.name} → {output_file.name}")
        print(f"📁 Output: {output_file}")
        print(f"\n📋 Next steps:")
        print(f"   1. Copy content from: {output_file}")
        print(f"   2. Paste into your ChatGPT conversation")
        print(f"   3. Or drag-and-drop the file into ChatGPT")
        
        return output_file
    
    def process_all(self) -> List[Path]:
        """Process all sync files"""
        if not SYNC_DIR.exists():
            print(f"❌ Sync directory not found: {SYNC_DIR}")
            return []
        
        md_files = list(SYNC_DIR.glob("*.md"))
        if not md_files:
            print("❌ No .md files found")
            return []
        
        output_files = []
        for md_file in sorted(md_files, key=lambda f: f.stat().st_mtime):
            output_file = self.process_sync_file(md_file)
            output_files.append(output_file)
            print(f"✅ Processed: {md_file.name} → {output_file.name}")
        
        print(f"\n📁 All files ready in: {OUTPUT_DIR}")
        print(f"📋 Upload to ChatGPT in chronological order for best context")
        
        return output_files

def main():
    """Main CLI interface"""
    import sys
    
    uploader = ChatGPTWebUploader()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python chatgpt_web_uploader.py --process-latest")
        print("  python chatgpt_web_uploader.py --process-all")
        return 1
    
    command = sys.argv[1]
    
    if command == "--process-latest":
        uploader.process_latest()
    elif command == "--process-all":
        uploader.process_all()
    else:
        print(f"❌ Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())