#!/usr/bin/env python3
"""
Setup ChatGPT GitHub Sync
=========================

Interactive setup for syncing your AI Memory Layer commits to ChatGPT.
"""

import os
import subprocess
from pathlib import Path

def create_env_file():
    """Create .env file with GitHub token"""
    print("🔑 GitHub Token Setup")
    print("=" * 50)
    
    print("\n📋 Steps to get your GitHub token:")
    print("1. Go to: https://github.com/settings/personal-access-tokens/new")
    print("2. Repository access: Select 'jeremy/ai-memory-layer' (or your repo)")
    print("3. Permissions needed:")
    print("   - Contents: Read")
    print("   - Metadata: Read")
    print("4. Generate token and copy it")
    
    # Get token from user
    token = input("\n🔑 Paste your GitHub token here: ").strip()
    
    if not token or not token.startswith(('ghp_', 'github_pat_')):
        print("❌ Invalid token format. Should start with 'ghp_' or 'github_pat_'")
        return False
    
    # Get repository
    repo = input("📁 Repository (e.g., jeremy/ai-memory-layer): ").strip()
    if not repo or '/' not in repo:
        print("❌ Invalid repository format. Should be 'username/repo-name'")
        return False
    
    # Create .env file
    env_content = f"""# ChatGPT GitHub Sync Configuration
GITHUB_TOKEN={token}
REPO={repo}
"""
    
    env_path = Path(".env.chatgpt")
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Created {env_path}")
    print("🔒 Token saved securely")
    
    return True

def test_connection():
    """Test GitHub API connection"""
    print("\n🔍 Testing GitHub connection...")
    
    try:
        result = subprocess.run([
            "python", "chatgpt_github_sync.py"
        ], capture_output=True, text=True, env={
            **os.environ,
            **dict(line.split('=', 1) for line in open('.env.chatgpt').read().strip().split('\n') if '=' in line)
        })
        
        if result.returncode == 0:
            print("✅ Connection successful!")
            print("📁 Files should be created in current directory")
            return True
        else:
            print("❌ Connection failed:")
            print(result.stderr)
            return False
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AI Memory Layer → ChatGPT Sync Setup")
    print("=" * 50)
    
    print("\nThis will help you sync your recent commits to ChatGPT for intelligent code analysis.")
    
    # Step 1: Create .env file
    if not create_env_file():
        return
    
    # Step 2: Test connection
    if not test_connection():
        print("\n❌ Setup incomplete. Please check your token and try again.")
        return
    
    # Step 3: Success instructions
    print("\n🎉 Setup Complete!")
    print("=" * 30)
    
    print("\n📋 Usage:")
    print("1. Run: python chatgpt_github_sync.py")
    print("2. Upload the generated .md file to ChatGPT")
    print("3. Ask ChatGPT about your code!")
    
    print("\n💡 Example ChatGPT prompts:")
    print("- 'Analyze my recent commits and summarize what's been built'")
    print("- 'What are the main components of the AI Memory Layer?'")
    print("- 'Review the latest changes and suggest improvements'")
    print("- 'Help me debug the webhook integration'")
    
    print("\n⚡ Automation:")
    print("- Add to your deploy script: python chatgpt_github_sync.py")
    print("- Set up a cron job for daily syncs")
    print("- Integrate with your CI/CD pipeline")

if __name__ == "__main__":
    main()