#!/usr/bin/env python3
"""
ChatGPT GitHub Sync
==================

Syncs your AI Memory Layer commits to ChatGPT using GitHub API.
This creates a persistent memory for ChatGPT about your project.
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatGPTGitHubSync:
    """Syncs GitHub repo data to ChatGPT"""
    
    def __init__(self, github_token: str, repo: str):
        self.github_token = github_token
        self.repo = repo
        self.github_api = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
    def get_recent_commits(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commits from GitHub API"""
        url = f"{self.github_api}/repos/{self.repo}/commits"
        params = {"per_page": limit}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch commits: {e}")
            return []
    
    def get_commit_details(self, commit_sha: str) -> Optional[Dict[str, Any]]:
        """Get detailed commit information including diffs"""
        url = f"{self.github_api}/repos/{self.repo}/commits/{commit_sha}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch commit {commit_sha}: {e}")
            return None
    
    def format_commit_for_chatgpt(self, commit_data: Dict[str, Any]) -> str:
        """Format commit data for ChatGPT upload"""
        sha = commit_data["sha"][:8]
        message = commit_data["commit"]["message"]
        author = commit_data["commit"]["author"]["name"]
        date = commit_data["commit"]["author"]["date"]
        
        # Format files changed
        files_changed = []
        for file in commit_data.get("files", []):
            status = file["status"]
            filename = file["filename"]
            additions = file.get("additions", 0)
            deletions = file.get("deletions", 0)
            files_changed.append(f"- {status}: {filename} (+{additions}/-{deletions})")
        
        # Format the commit summary
        summary = f"""# 🔄 AI Memory Layer Commit: {sha}

## 📊 Commit Information
- **SHA**: `{commit_data["sha"]}`
- **Author**: {author}
- **Date**: {date}
- **Message**: {message}

## 📁 Files Changed ({len(files_changed)} files):
{chr(10).join(files_changed)}

## 🔍 Key Changes:
"""
        
        # Add diff snippets for key files
        important_files = [f for f in commit_data.get("files", []) if 
                          f["filename"].endswith(('.py', '.js', '.ts', '.md'))]
        
        for file in important_files[:5]:  # Limit to 5 files
            if file.get("patch"):
                summary += f"""
### {file["filename"]}
```diff
{file["patch"][:1000]}{"..." if len(file["patch"]) > 1000 else ""}
```
"""
        
        summary += f"""
## 🧠 Context for AI Assistant:
This commit is part of the AI Memory Layer project - a Python-based system for semantic memory storage and retrieval using FAISS vector storage and OpenAI embeddings. The system includes:

- FAISS-based vector storage for efficient similarity search
- OpenAI embeddings integration (text-embedding-3-small)
- FastAPI REST API with comprehensive endpoints
- GitHub webhook integration for automatic updates
- Real-time memory indexing and retrieval capabilities

**Project Status**: This is commit {sha} in the development of an intelligent memory system that can store, index, and retrieve information using semantic search.
"""
        
        return summary
    
    def create_chatgpt_upload_file(self, commits: List[Dict[str, Any]], output_file: str = "chatgpt_commits.md"):
        """Create a file ready for ChatGPT upload"""
        content = [f"""# AI Memory Layer - Recent Commits
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Repository: {self.repo}

This document contains recent commits from the AI Memory Layer project for ChatGPT to understand the codebase evolution.

"""]
        
        for i, commit in enumerate(commits, 1):
            content.append(f"\n{'='*80}")
            content.append(f"COMMIT {i}/{len(commits)}")
            content.append('='*80)
            content.append(self.format_commit_for_chatgpt(commit))
        
        # Add project context at the end
        content.append(f"""
{'='*80}
# PROJECT CONTEXT SUMMARY
{'='*80}

## About AI Memory Layer
The AI Memory Layer is a sophisticated Python system that provides:

1. **Vector Storage**: FAISS-based efficient similarity search
2. **Embeddings**: OpenAI text-embedding-3-small integration  
3. **API**: FastAPI REST endpoints for memory operations
4. **Automation**: GitHub webhook integration
5. **Real-time Processing**: Live memory indexing and retrieval

## Key Components:
- `core/memory_engine.py` - Central memory management
- `storage/faiss_store.py` - Vector storage implementation
- `integrations/openai_integration.py` - OpenAI API wrapper
- `api/main.py` - FastAPI REST API
- `tests/` - Comprehensive test suite

## Current Focus:
The recent commits show active development in webhook automation, ChatGPT integration, and performance optimization. The system is designed to maintain conversation context and improve AI responses through intelligent memory storage and retrieval.

**For ChatGPT**: Use this information to understand the codebase, answer questions about recent changes, suggest improvements, and help with debugging or development tasks.
""")
        
        full_content = "\n".join(content)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        logger.info(f"Created ChatGPT upload file: {output_file}")
        logger.info(f"File size: {len(full_content):,} characters")
        
        return output_file
    
    def create_json_export(self, commits: List[Dict[str, Any]], output_file: str = "commits_data.json"):
        """Create JSON export of commit data"""
        export_data = {
            "repository": self.repo,
            "exported_at": datetime.now().isoformat(),
            "commits": []
        }
        
        for commit in commits:
            commit_info = {
                "sha": commit["sha"],
                "message": commit["commit"]["message"],
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
                "url": commit["html_url"],
                "files_changed": len(commit.get("files", [])),
                "additions": sum(f.get("additions", 0) for f in commit.get("files", [])),
                "deletions": sum(f.get("deletions", 0) for f in commit.get("files", [])),
                "files": [
                    {
                        "filename": f["filename"],
                        "status": f["status"],
                        "additions": f.get("additions", 0),
                        "deletions": f.get("deletions", 0)
                    }
                    for f in commit.get("files", [])
                ]
            }
            export_data["commits"].append(commit_info)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Created JSON export: {output_file}")
        return output_file

def main():
    """Main function to sync GitHub commits to ChatGPT format"""
    # Configuration
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    REPO = os.getenv("REPO", "jeremy/ai-memory-layer")  # Update with your repo
    
    if not GITHUB_TOKEN:
        print("❌ GITHUB_TOKEN environment variable required")
        print("\n📋 Setup Instructions:")
        print("1. Go to GitHub > Settings > Developer Settings > Personal Access Tokens")
        print("2. Generate a fine-grained token with 'Contents: Read' permission")
        print("3. Set environment variable: export GITHUB_TOKEN='your_token_here'")
        print("4. Set repository: export REPO='yourusername/ai-memory-layer'")
        return
    
    # Initialize sync
    syncer = ChatGPTGitHubSync(GITHUB_TOKEN, REPO)
    
    print(f"🚀 Syncing recent commits from {REPO}...")
    
    # Get recent commits
    commits = syncer.get_recent_commits(limit=10)
    if not commits:
        print("❌ No commits found or API error")
        return
    
    print(f"📁 Found {len(commits)} recent commits")
    
    # Get detailed information for each commit
    detailed_commits = []
    for commit in commits:
        print(f"🔍 Processing commit {commit['sha'][:8]}...")
        details = syncer.get_commit_details(commit['sha'])
        if details:
            detailed_commits.append(details)
    
    if not detailed_commits:
        print("❌ No detailed commit data retrieved")
        return
    
    # Create ChatGPT upload file
    chatgpt_file = syncer.create_chatgpt_upload_file(detailed_commits)
    json_file = syncer.create_json_export(detailed_commits)
    
    print(f"\n✅ Files created:")
    print(f"📄 ChatGPT upload: {chatgpt_file}")
    print(f"📊 JSON data: {json_file}")
    
    print(f"\n📋 Next Steps:")
    print(f"1. Upload {chatgpt_file} to ChatGPT")
    print(f"2. Ask ChatGPT: 'Analyze these recent commits and tell me about the AI Memory Layer project progress'")
    print(f"3. Use ChatGPT to ask specific questions about your code changes")
    
    # Show preview
    with open(chatgpt_file, 'r') as f:
        preview = f.read()[:500]
    
    print(f"\n📖 Preview of {chatgpt_file}:")
    print("=" * 50)
    print(preview + "...")
    print("=" * 50)

if __name__ == "__main__":
    main()