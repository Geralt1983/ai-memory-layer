#!/usr/bin/env python3
"""
GitHub Webhook Setup Script
============================

Automatically configures GitHub webhook for ChatGPT sync integration.
This script uses the GitHub API to create a webhook that will trigger
on push events and send them to your webhook receiver.

Usage:
    python setup_github_webhook.py

Environment Variables:
    GITHUB_TOKEN    - GitHub Personal Access Token with repo access
    REPO_OWNER      - GitHub repository owner/username  
    REPO_NAME       - GitHub repository name
    WEBHOOK_URL     - Your webhook receiver URL (e.g., http://your-server:8001/webhook)
    WEBHOOK_SECRET  - Secret for webhook security (optional but recommended)

Example:
    export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
    export REPO_OWNER="your-username"
    export REPO_NAME="ai-memory-layer"
    export WEBHOOK_URL="http://18.224.179.36:8001/webhook"
    export WEBHOOK_SECRET="your-webhook-secret"
    python setup_github_webhook.py
"""

import os
import sys
import requests
import json
from typing import Dict, Any, Optional

def create_github_webhook(
    token: str,
    owner: str, 
    repo: str,
    webhook_url: str,
    secret: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a GitHub webhook using the GitHub API
    
    Args:
        token: GitHub Personal Access Token
        owner: Repository owner/username
        repo: Repository name
        webhook_url: URL to receive webhook events
        secret: Optional webhook secret for security
        
    Returns:
        Response from GitHub API
    """
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    # Webhook configuration
    webhook_config = {
        "name": "web",
        "active": True,
        "events": ["push"],  # Only listen for push events
        "config": {
            "url": webhook_url,
            "content_type": "json",
            "insecure_ssl": "0"  # Require SSL
        }
    }
    
    # Add secret if provided
    if secret:
        webhook_config["config"]["secret"] = secret
    
    # GitHub API endpoint
    api_url = f"https://api.github.com/repos/{owner}/{repo}/hooks"
    
    try:
        response = requests.post(
            api_url, 
            headers=headers, 
            json=webhook_config,
            timeout=30
        )
        
        if response.status_code == 201:
            return {
                "success": True,
                "webhook": response.json(),
                "message": "Webhook created successfully"
            }
        else:
            return {
                "success": False,
                "error": response.json() if response.text else "Unknown error",
                "status_code": response.status_code,
                "message": f"Failed to create webhook: {response.status_code}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Request failed: {e}"
        }

def list_existing_webhooks(token: str, owner: str, repo: str) -> Dict[str, Any]:
    """List existing webhooks for the repository"""
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/hooks"
    
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            webhooks = response.json()
            return {
                "success": True,
                "webhooks": webhooks,
                "count": len(webhooks)
            }
        else:
            return {
                "success": False,
                "error": response.json() if response.text else "Unknown error",
                "status_code": response.status_code
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e)
        }

def delete_webhook(token: str, owner: str, repo: str, webhook_id: int) -> Dict[str, Any]:
    """Delete a specific webhook"""
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json", 
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/hooks/{webhook_id}"
    
    try:
        response = requests.delete(api_url, headers=headers, timeout=30)
        
        if response.status_code == 204:
            return {
                "success": True,
                "message": f"Webhook {webhook_id} deleted successfully"
            }
        else:
            return {
                "success": False,
                "error": response.json() if response.text else "Unknown error",
                "status_code": response.status_code
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e)
        }

def test_webhook_connectivity(webhook_url: str) -> Dict[str, Any]:
    """Test if the webhook receiver is accessible"""
    
    try:
        # Try to reach the root endpoint first
        base_url = webhook_url.replace('/webhook', '')
        response = requests.get(base_url, timeout=10)
        
        if response.status_code == 200:
            return {
                "success": True,
                "message": "Webhook receiver is accessible",
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "message": f"Webhook receiver returned {response.status_code}",
                "status_code": response.status_code
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Cannot reach webhook receiver: {e}",
            "error": str(e)
        }

def main():
    """Main setup function"""
    
    print("🔗 GitHub → ChatGPT Webhook Setup")
    print("=" * 40)
    
    # Get configuration from environment
    token = os.getenv("GITHUB_TOKEN")
    owner = os.getenv("REPO_OWNER")
    repo = os.getenv("REPO_NAME") 
    webhook_url = os.getenv("WEBHOOK_URL")
    secret = os.getenv("WEBHOOK_SECRET")
    
    # Validate required configuration
    if not all([token, owner, repo, webhook_url]):
        print("❌ Missing required environment variables:")
        if not token:
            print("   - GITHUB_TOKEN: GitHub Personal Access Token")
        if not owner:
            print("   - REPO_OWNER: Repository owner/username")
        if not repo:
            print("   - REPO_NAME: Repository name")
        if not webhook_url:
            print("   - WEBHOOK_URL: Webhook receiver URL")
        
        print("\nExample:")
        print('export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"')
        print('export REPO_OWNER="your-username"')
        print('export REPO_NAME="ai-memory-layer"')
        print('export WEBHOOK_URL="http://18.224.179.36:8001/webhook"')
        print('export WEBHOOK_SECRET="your-secret"  # Optional')
        
        sys.exit(1)
    
    print(f"📂 Repository: {owner}/{repo}")
    print(f"🔗 Webhook URL: {webhook_url}")
    print(f"🔐 Secret configured: {'Yes' if secret else 'No'}")
    print()
    
    # Test webhook connectivity
    print("🧪 Testing webhook receiver connectivity...")
    connectivity_result = test_webhook_connectivity(webhook_url)
    if connectivity_result["success"]:
        print(f"✅ {connectivity_result['message']}")
    else:
        print(f"⚠️  {connectivity_result['message']}")
        print("   Make sure your webhook receiver is running and accessible")
    print()
    
    # List existing webhooks
    print("🔍 Checking existing webhooks...")
    list_result = list_existing_webhooks(token, owner, repo)
    
    if list_result["success"]:
        webhooks = list_result["webhooks"]
        print(f"📋 Found {len(webhooks)} existing webhook(s)")
        
        # Check for duplicates
        for i, webhook in enumerate(webhooks):
            webhook_config_url = webhook.get("config", {}).get("url", "")
            print(f"   {i+1}. ID: {webhook['id']}, URL: {webhook_config_url}")
            
            if webhook_config_url == webhook_url:
                print(f"   ⚠️  Duplicate found! Webhook {webhook['id']} already points to {webhook_url}")
                
                choice = input("   Delete existing webhook? (y/N): ").lower().strip()
                if choice == 'y':
                    delete_result = delete_webhook(token, owner, repo, webhook['id'])
                    if delete_result["success"]:
                        print(f"   ✅ {delete_result['message']}")
                    else:
                        print(f"   ❌ Failed to delete: {delete_result.get('error', 'Unknown error')}")
                else:
                    print("   Keeping existing webhook. Skipping creation.")
                    sys.exit(0)
    else:
        print(f"❌ Failed to list webhooks: {list_result.get('error', 'Unknown error')}")
    
    print()
    
    # Create the webhook
    print("🔨 Creating GitHub webhook...")
    result = create_github_webhook(token, owner, repo, webhook_url, secret)
    
    if result["success"]:
        webhook = result["webhook"]
        print("✅ Webhook created successfully!")
        print(f"   ID: {webhook['id']}")
        print(f"   URL: {webhook['config']['url']}")
        print(f"   Events: {', '.join(webhook['events'])}")
        print(f"   Active: {webhook['active']}")
        
        # Test the webhook
        print("\n🧪 To test your webhook:")
        print("   1. Make a commit and push to your repository")
        print("   2. Check your webhook receiver logs")
        print("   3. Look for generated files in the chatgpt-sync directory")
        
        print(f"\n📁 ChatGPT sync files will be saved to: ./chatgpt-sync/")
        print("   Upload these files to ChatGPT for code review and discussion")
        
    else:
        print(f"❌ Failed to create webhook: {result['message']}")
        if 'error' in result:
            print(f"   Error details: {result['error']}")
        
        if result.get('status_code') == 401:
            print("   Check your GitHub token permissions (needs repo access)")
        elif result.get('status_code') == 404:
            print("   Check repository owner/name or token access to private repos")

if __name__ == "__main__":
    main()