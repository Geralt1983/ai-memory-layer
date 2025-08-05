#!/usr/bin/env python3
"""
ChatGPT API Auto-Uploader
==========================

Automatically uploads GitHub webhook sync files to ChatGPT via OpenAI API.
This completes the full automation: GitHub → Webhook → ChatGPT API integration.

Features:
- Monitors chatgpt-sync directory for new .md files
- Uploads files to OpenAI Files API
- Creates ChatGPT conversations with commit context
- Maintains conversation threads for project continuity
- Supports both one-time uploads and continuous monitoring

Usage:
    python chatgpt_api_uploader.py --upload-latest
    python chatgpt_api_uploader.py --monitor
    python chatgpt_api_uploader.py --upload-all

Environment Variables:
    OPENAI_API_KEY          - OpenAI API key for ChatGPT access
    CHATGPT_ASSISTANT_ID    - Optional: Specific assistant ID to use
    CHATGPT_THREAD_ID       - Optional: Existing thread to continue
    SYNC_DIR                - Directory with .md files (default: ./chatgpt-sync)
    AUTO_UPLOAD_ENABLED     - Enable automatic uploads (default: false)
"""

import os
import time
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import hashlib

import openai
from openai import OpenAI
import requests

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHATGPT_ASSISTANT_ID = os.getenv("CHATGPT_ASSISTANT_ID", "")
CHATGPT_THREAD_ID = os.getenv("CHATGPT_THREAD_ID", "")
SYNC_DIR = Path(os.getenv("SYNC_DIR", "./chatgpt-sync"))
AUTO_UPLOAD_ENABLED = os.getenv("AUTO_UPLOAD_ENABLED", "false").lower() == "true"
UPLOAD_LOG_FILE = SYNC_DIR / "upload_log.json"

# Initialize OpenAI client
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY environment variable is required")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

@dataclass
class UploadRecord:
    """Track uploaded files to avoid duplicates"""
    filename: str
    file_id: str
    uploaded_at: str
    commit_sha: str
    thread_id: Optional[str] = None
    message_id: Optional[str] = None

class ChatGPTUploader:
    """Handles uploading sync files to ChatGPT via OpenAI API"""
    
    def __init__(self):
        self.upload_log = self.load_upload_log()
        self.client = client
        
    def load_upload_log(self) -> Dict[str, UploadRecord]:
        """Load previously uploaded files log"""
        if UPLOAD_LOG_FILE.exists():
            try:
                with open(UPLOAD_LOG_FILE, 'r') as f:
                    data = json.load(f)
                    return {k: UploadRecord(**v) for k, v in data.items()}
            except Exception as e:
                print(f"⚠️  Failed to load upload log: {e}")
        return {}
    
    def save_upload_log(self):
        """Save upload log to file"""
        try:
            SYNC_DIR.mkdir(exist_ok=True)
            data = {k: v.__dict__ for k, v in self.upload_log.items()}
            with open(UPLOAD_LOG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save upload log: {e}")
    
    def get_file_hash(self, filepath: Path) -> str:
        """Get file content hash to detect changes"""
        return hashlib.md5(filepath.read_bytes()).hexdigest()
    
    def upload_file_to_openai(self, filepath: Path) -> Optional[str]:
        """Upload file to OpenAI Files API"""
        try:
            with open(filepath, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='assistants'  # Use 'assistants' for ChatGPT integration
                )
            return response.id
        except Exception as e:
            print(f"❌ Failed to upload {filepath.name}: {e}")
            return None
    
    def create_chatgpt_message(self, file_id: str, commit_info: Dict[str, Any]) -> Optional[str]:
        """Create a ChatGPT message with the uploaded file"""
        try:
            # Create or use existing thread
            thread_id = CHATGPT_THREAD_ID
            if not thread_id:
                thread = self.client.beta.threads.create()
                thread_id = thread.id
                print(f"📝 Created new ChatGPT thread: {thread_id}")
            
            # Create message with file attachment
            message_content = f"""🔄 **GitHub Commit Update - AI Memory Layer**

New commit from repository: {commit_info.get('commit_sha', 'unknown')}
Author: {commit_info.get('author', 'unknown')}
Message: {commit_info.get('message', 'unknown')}

Please review the attached commit details and code changes. Let me know if you have any questions, suggestions, or if you'd like me to analyze specific aspects of the changes.

The attached file contains:
- Complete commit information
- All changed files with full content
- File-by-file breakdown of modifications

Ready for your feedback and next steps!"""

            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message_content,
                file_ids=[file_id]
            )
            
            print(f"✅ Created ChatGPT message: {message.id}")
            print(f"🔗 Thread: https://chat.openai.com/chat/{thread_id}")
            
            return message.id
            
        except Exception as e:
            print(f"❌ Failed to create ChatGPT message: {e}")
            return None
    
    def parse_commit_info(self, md_content: str) -> Dict[str, Any]:
        """Extract commit information from markdown content"""
        info = {}
        lines = md_content.split('\n')
        
        for line in lines:
            if line.startswith('- **SHA**:'):
                info['commit_sha'] = line.split('`')[1] if '`' in line else 'unknown'
            elif line.startswith('- **Author**:'):
                info['author'] = line.split(': ', 1)[1] if ': ' in line else 'unknown'
            elif line.startswith('- **Message**:'):
                info['message'] = line.split(': ', 1)[1] if ': ' in line else 'unknown'
            elif line.startswith('- **Time**:'):
                info['timestamp'] = line.split(': ', 1)[1] if ': ' in line else 'unknown'
        
        return info
    
    def upload_sync_file(self, filepath: Path) -> bool:
        """Upload a single sync file to ChatGPT"""
        try:
            # Check if already uploaded
            file_hash = self.get_file_hash(filepath)
            if filepath.name in self.upload_log:
                existing_record = self.upload_log[filepath.name]
                if hasattr(existing_record, 'file_hash') and existing_record.file_hash == file_hash:
                    print(f"⏭️  Skipping {filepath.name} (already uploaded)")
                    return True
            
            print(f"📤 Uploading {filepath.name} to ChatGPT...")
            
            # Read and parse commit info
            md_content = filepath.read_text(encoding='utf-8')
            commit_info = self.parse_commit_info(md_content)
            
            # Upload file to OpenAI
            file_id = self.upload_file_to_openai(filepath)
            if not file_id:
                return False
            
            # Create ChatGPT message
            message_id = self.create_chatgpt_message(file_id, commit_info)
            
            # Record successful upload
            record = UploadRecord(
                filename=filepath.name,
                file_id=file_id,
                uploaded_at=datetime.now().isoformat(),
                commit_sha=commit_info.get('commit_sha', 'unknown'),
                thread_id=CHATGPT_THREAD_ID,
                message_id=message_id
            )
            record.file_hash = file_hash
            
            self.upload_log[filepath.name] = record
            self.save_upload_log()
            
            print(f"✅ Successfully uploaded {filepath.name}")
            print(f"   File ID: {file_id}")
            print(f"   Message ID: {message_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload {filepath.name}: {e}")
            return False
    
    def upload_latest_file(self) -> bool:
        """Upload the most recent .md file"""
        if not SYNC_DIR.exists():
            print(f"❌ Sync directory does not exist: {SYNC_DIR}")
            return False
        
        md_files = list(SYNC_DIR.glob("*.md"))
        if not md_files:
            print("❌ No .md files found in sync directory")
            return False
        
        # Get the most recent file
        latest_file = max(md_files, key=lambda f: f.stat().st_mtime)
        print(f"📋 Latest file: {latest_file.name}")
        
        return self.upload_sync_file(latest_file)
    
    def upload_all_files(self) -> int:
        """Upload all .md files that haven't been uploaded yet"""
        if not SYNC_DIR.exists():
            print(f"❌ Sync directory does not exist: {SYNC_DIR}")
            return 0
        
        md_files = list(SYNC_DIR.glob("*.md"))
        if not md_files:
            print("❌ No .md files found in sync directory")
            return 0
        
        uploaded_count = 0
        for md_file in sorted(md_files, key=lambda f: f.stat().st_mtime):
            if self.upload_sync_file(md_file):
                uploaded_count += 1
            time.sleep(1)  # Rate limiting
        
        return uploaded_count
    
    def monitor_directory(self, check_interval: int = 30):
        """Monitor sync directory for new files and auto-upload"""
        print(f"👀 Monitoring {SYNC_DIR} for new files (checking every {check_interval}s)")
        print("   Press Ctrl+C to stop")
        
        known_files = set()
        if SYNC_DIR.exists():
            known_files = {f.name for f in SYNC_DIR.glob("*.md")}
        
        try:
            while True:
                if SYNC_DIR.exists():
                    current_files = {f.name for f in SYNC_DIR.glob("*.md")}
                    new_files = current_files - known_files
                    
                    for new_file in new_files:
                        filepath = SYNC_DIR / new_file
                        print(f"🆕 Detected new file: {new_file}")
                        if AUTO_UPLOAD_ENABLED:
                            self.upload_sync_file(filepath)
                        else:
                            print(f"   Auto-upload disabled. Run with --upload-latest to upload.")
                    
                    known_files = current_files
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")

def main():
    """Main CLI interface"""
    import sys
    
    print("🤖 ChatGPT API Auto-Uploader")
    print("=" * 40)
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY environment variable is required")
        print("\nSet it with:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    uploader = ChatGPTUploader()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python chatgpt_api_uploader.py --upload-latest")
        print("  python chatgpt_api_uploader.py --upload-all")
        print("  python chatgpt_api_uploader.py --monitor")
        print("  python chatgpt_api_uploader.py --status")
        return 1
    
    command = sys.argv[1]
    
    if command == "--upload-latest":
        print("📤 Uploading latest sync file...")
        success = uploader.upload_latest_file()
        return 0 if success else 1
        
    elif command == "--upload-all":
        print("📤 Uploading all sync files...")
        count = uploader.upload_all_files()
        print(f"✅ Uploaded {count} files")
        return 0
        
    elif command == "--monitor":
        uploader.monitor_directory()
        return 0
        
    elif command == "--status":
        print(f"📊 Upload Status:")
        print(f"   Sync directory: {SYNC_DIR}")
        print(f"   Auto-upload enabled: {AUTO_UPLOAD_ENABLED}")
        print(f"   OpenAI API configured: {bool(OPENAI_API_KEY)}")
        print(f"   Files uploaded: {len(uploader.upload_log)}")
        
        if SYNC_DIR.exists():
            md_files = list(SYNC_DIR.glob("*.md"))
            print(f"   Available .md files: {len(md_files)}")
            
            if md_files:
                latest = max(md_files, key=lambda f: f.stat().st_mtime)
                print(f"   Latest file: {latest.name}")
        
        return 0
        
    else:
        print(f"❌ Unknown command: {command}")
        return 1

if __name__ == "__main__":
    exit(main())