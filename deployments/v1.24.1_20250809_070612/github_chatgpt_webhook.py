#!/usr/bin/env python3
"""
GitHub → ChatGPT Webhook Sync Server
=====================================

Receives GitHub webhook events and syncs code changes to ChatGPT-compatible format.
This enables semi-automated code review and discussion in ChatGPT conversations.

Features:
- Receives GitHub push events via webhook
- Downloads changed files from private repos (with PAT authentication)
- Generates ChatGPT-compatible file summaries and diffs
- Saves synced content for easy upload to ChatGPT
- Includes commit messages and context for better AI understanding
- Supports filtering by file types and directories

Usage:
    python github_chatgpt_webhook.py

Environment Variables:
    GITHUB_TOKEN        - GitHub Personal Access Token for private repos
    WEBHOOK_SECRET      - GitHub webhook secret for security (optional)
    SYNC_DIR            - Directory to save ChatGPT sync files (default: ./chatgpt-sync)
    REPO_OWNER          - GitHub repository owner/username
    REPO_NAME           - GitHub repository name
    PORT                - Server port (default: 8001)
"""

import os
import json
import hmac
import hashlib
import requests
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables from .env.webhook file
try:
    from dotenv import load_dotenv
    load_dotenv(".env.webhook")
except ImportError:
    print("python-dotenv not available, using system environment variables")

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
SYNC_DIR = Path(os.getenv("SYNC_DIR", "./chatgpt-sync"))
REPO_OWNER = os.getenv("REPO_OWNER", "")
REPO_NAME = os.getenv("REPO_NAME", "")
PORT = int(os.getenv("PORT", "8001"))

# File type filters for relevant code changes
RELEVANT_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', 
    '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.cs',
    '.md', '.txt', '.yml', '.yaml', '.json', '.toml', '.cfg', '.ini'
}

# Directory filters (ignore these)
IGNORE_DIRS = {
    '__pycache__', 'node_modules', '.git', '.pytest_cache', 'venv', 
    'env', '.env', 'dist', 'build', '.next', 'target'
}

app = FastAPI(title="GitHub → ChatGPT Webhook Sync", version="1.0.0")

@dataclass
class FileChange:
    """Represents a file change in a commit"""
    filename: str
    status: str  # added, modified, removed
    additions: int
    deletions: int
    changes: int
    patch: Optional[str] = None
    content: Optional[str] = None
    previous_content: Optional[str] = None

@dataclass
class CommitInfo:
    """Represents commit information for ChatGPT context"""
    sha: str
    message: str
    author: str
    timestamp: str
    files_changed: List[FileChange]
    additions: int
    deletions: int
    url: str

def verify_github_signature(payload_body: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature for security"""
    if not WEBHOOK_SECRET:
        return True  # Skip verification if no secret configured
    
    expected_signature = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(),
        payload_body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_signature, signature)

def is_relevant_file(filepath: str) -> bool:
    """Check if file should be synced to ChatGPT"""
    path = Path(filepath)
    
    # Check if in ignored directory
    for part in path.parts:
        if part in IGNORE_DIRS:
            return False
    
    # Check file extension
    return path.suffix.lower() in RELEVANT_EXTENSIONS

def get_file_content(owner: str, repo: str, filepath: str, ref: str = "main") -> Optional[str]:
    """Download file content from GitHub API"""
    if not GITHUB_TOKEN:
        print(f"Warning: No GitHub token configured, cannot fetch {filepath}")
        return None
    
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.raw"
    }
    
    params = {"ref": ref}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch {filepath}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {filepath}: {e}")
        return None

def parse_webhook_payload(payload: Dict[str, Any]) -> Optional[CommitInfo]:
    """Parse GitHub webhook payload into structured commit info"""
    try:
        # Handle push events
        if "commits" not in payload or not payload["commits"]:
            return None
        
        # Get the latest commit (head commit)
        commit = payload["head_commit"]
        repo = payload["repository"]
        
        # Parse commit details
        commit_info = CommitInfo(
            sha=commit["id"],
            message=commit["message"],
            author=commit["author"]["name"],
            timestamp=commit["timestamp"],
            files_changed=[],
            additions=0,
            deletions=0,
            url=commit["url"]
        )
        
        # Process changed files
        all_files = set(commit.get("added", []) + commit.get("modified", []) + commit.get("removed", []))
        
        for filename in all_files:
            if not is_relevant_file(filename):
                continue
            
            # Determine file status
            status = "modified"
            if filename in commit.get("added", []):
                status = "added"
            elif filename in commit.get("removed", []):
                status = "removed"
            
            # Get file content
            content = None
            if status != "removed":
                content = get_file_content(
                    repo["owner"]["name"], 
                    repo["name"], 
                    filename, 
                    commit["id"]
                )
            
            file_change = FileChange(
                filename=filename,
                status=status,
                additions=0,  # GitHub doesn't provide this in push events
                deletions=0,
                changes=0,
                content=content
            )
            
            commit_info.files_changed.append(file_change)
        
        return commit_info
        
    except Exception as e:
        print(f"Error parsing webhook payload: {e}")
        return None

def generate_chatgpt_summary(commit_info: CommitInfo) -> str:
    """Generate a ChatGPT-friendly summary of the commit"""
    timestamp = datetime.fromisoformat(commit_info.timestamp.replace('Z', '+00:00'))
    
    summary = f"""# 🔄 Git Commit Update - AI Memory Layer

## 📊 Commit Information
- **SHA**: `{commit_info.sha[:8]}`
- **Author**: {commit_info.author}
- **Time**: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Message**: {commit_info.message}
- **Files Changed**: {len(commit_info.files_changed)}

## 📁 Changed Files

"""
    
    for file_change in commit_info.files_changed:
        status_emoji = {"added": "➕", "modified": "✏️", "removed": "❌"}.get(file_change.status, "📄")
        summary += f"### {status_emoji} `{file_change.filename}` ({file_change.status})\n\n"
        
        if file_change.content and file_change.status != "removed":
            # Truncate very long files
            content = file_change.content
            if len(content) > 2000:
                content = content[:2000] + f"\n\n... (truncated, full file has {len(file_change.content)} characters)"
            
            summary += f"```{Path(file_change.filename).suffix[1:] if Path(file_change.filename).suffix else 'text'}\n"
            summary += content
            summary += "\n```\n\n"
        elif file_change.status == "removed":
            summary += "*File was removed*\n\n"
        else:
            summary += "*Could not fetch file content*\n\n"
    
    summary += f"""
---
*Auto-generated by GitHub → ChatGPT Webhook Sync*  
*Commit URL*: {commit_info.url}
"""
    
    return summary

def save_sync_file(commit_info: CommitInfo, summary: str) -> Path:
    """Save the ChatGPT summary to a file"""
    SYNC_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.fromisoformat(commit_info.timestamp.replace('Z', '+00:00'))
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{commit_info.sha[:8]}.md"
    filepath = SYNC_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Also save raw commit info as JSON
    json_filepath = SYNC_DIR / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{commit_info.sha[:8]}.json"
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(asdict(commit_info), f, indent=2)
    
    return filepath

async def sync_to_chatgpt(commit_info: CommitInfo):
    """Background task to sync commit to ChatGPT format"""
    try:
        print(f"🔄 Processing commit {commit_info.sha[:8]} by {commit_info.author}")
        
        # Generate ChatGPT summary
        summary = generate_chatgpt_summary(commit_info)
        
        # Save to file
        filepath = save_sync_file(commit_info, summary)
        
        print(f"✅ Saved ChatGPT sync file: {filepath}")
        print(f"📁 Files changed: {len(commit_info.files_changed)}")
        
        # Print summary for immediate visibility
        print("\n" + "="*60)
        print("CHATGPT SYNC SUMMARY")
        print("="*60)
        print(summary[:500] + "..." if len(summary) > 500 else summary)
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error syncing to ChatGPT: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "GitHub → ChatGPT Webhook Sync",
        "status": "running",
        "sync_dir": str(SYNC_DIR),
        "github_configured": bool(GITHUB_TOKEN),
        "webhook_secured": bool(WEBHOOK_SECRET)
    }

@app.get("/sync-files")
async def list_sync_files():
    """List generated ChatGPT sync files"""
    if not SYNC_DIR.exists():
        return {"files": []}
    
    files = []
    for file in SYNC_DIR.glob("*.md"):
        stat = file.stat()
        files.append({
            "filename": file.name,
            "path": str(file),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    files.sort(key=lambda x: x["modified"], reverse=True)
    return {"files": files}

@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle GitHub webhook events"""
    try:
        # Get raw payload
        payload_body = await request.body()
        
        # Verify signature if configured
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not verify_github_signature(payload_body, signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse JSON payload
        try:
            payload = json.loads(payload_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # Check if it's a push event
        event_type = request.headers.get("X-GitHub-Event")
        if event_type != "push":
            return {"message": f"Ignored event type: {event_type}"}
        
        # Parse commit information
        commit_info = parse_webhook_payload(payload)
        if not commit_info:
            return {"message": "No relevant commits found"}
        
        # Background task to sync to ChatGPT
        background_tasks.add_task(sync_to_chatgpt, commit_info)
        
        return {
            "message": "Webhook received successfully",
            "commit": commit_info.sha[:8],
            "files_changed": len(commit_info.files_changed),
            "author": commit_info.author
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting GitHub → ChatGPT Webhook Sync Server")
    print(f"📁 Sync directory: {SYNC_DIR}")
    print(f"🔑 GitHub token configured: {bool(GITHUB_TOKEN)}")
    print(f"🔒 Webhook secret configured: {bool(WEBHOOK_SECRET)}")
    print(f"📡 Listening on port {PORT}")
    
    # Create sync directory
    SYNC_DIR.mkdir(exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)