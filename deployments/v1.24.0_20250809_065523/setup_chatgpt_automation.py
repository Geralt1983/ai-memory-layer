#!/usr/bin/env python3
"""
ChatGPT Automation Setup Script
================================

One-command setup for complete GitHub → ChatGPT automation.
This script configures the entire pipeline for automatic code-to-conversation sync.

Features:
- Tests OpenAI API connectivity
- Creates ChatGPT thread for project conversations
- Configures auto-upload service
- Updates webhook receiver with ChatGPT integration
- Provides complete setup verification

Usage:
    python setup_chatgpt_automation.py

Environment Variables Required:
    OPENAI_API_KEY          - OpenAI API key from https://platform.openai.com/api-keys
    
Optional:
    CHATGPT_THREAD_ID       - Existing thread ID to continue conversations
    PROJECT_NAME            - Project name for ChatGPT context (default: AI Memory Layer)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("❌ OpenAI library not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHATGPT_THREAD_ID = os.getenv("CHATGPT_THREAD_ID", "")
PROJECT_NAME = os.getenv("PROJECT_NAME", "AI Memory Layer")
SYNC_DIR = Path("./chatgpt-sync")
CONFIG_FILE = Path(".env.chatgpt")

class ChatGPTAutomationSetup:
    """Setup and configure ChatGPT automation"""
    
    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.config = {}
    
    def test_openai_connection(self) -> bool:
        """Test OpenAI API connectivity"""
        try:
            print("🧪 Testing OpenAI API connection...")
            
            # Test with a simple API call
            models = self.client.models.list()
            model_names = [model.id for model in models.data[:5]]
            
            print(f"✅ OpenAI API connection successful")
            print(f"   Available models: {', '.join(model_names)}")
            return True
            
        except Exception as e:
            print(f"❌ OpenAI API connection failed: {e}")
            return False
    
    def create_or_get_thread(self) -> Optional[str]:
        """Create a new ChatGPT thread or use existing one"""
        try:
            if CHATGPT_THREAD_ID:
                print(f"🔗 Using existing ChatGPT thread: {CHATGPT_THREAD_ID}")
                # Verify thread exists
                thread = self.client.beta.threads.retrieve(CHATGPT_THREAD_ID)
                return thread.id
            else:
                print("🆕 Creating new ChatGPT thread...")
                thread = self.client.beta.threads.create()
                print(f"✅ Created ChatGPT thread: {thread.id}")
                
                # Send initial project context message
                self.send_initial_context(thread.id)
                
                return thread.id
                
        except Exception as e:
            print(f"❌ Failed to create/get ChatGPT thread: {e}")
            return None
    
    def send_initial_context(self, thread_id: str):
        """Send initial project context to ChatGPT thread"""
        try:
            context_message = f"""🚀 **{PROJECT_NAME} - Automated Code Sync**

This thread will receive automated updates from your {PROJECT_NAME} repository via GitHub webhooks.

**What you'll see here:**
- 🔄 Automatic commit summaries with full code changes
- 📁 Complete file content for every modified file  
- 📊 Commit metadata (author, timestamp, message)
- 🔗 Direct links to GitHub commits

**How it works:**
1. You push code to GitHub
2. GitHub webhook triggers file generation
3. Files are automatically uploaded to this ChatGPT conversation
4. You get instant AI code review and assistance

**Getting Started:**
- Push any commit to your repository to see the automation in action
- Ask me to review specific changes or suggest improvements
- Request refactoring, debugging help, or architecture advice
- I'll maintain full context of your project evolution

Ready to assist with your {PROJECT_NAME} development! 🤖✨"""

            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user", 
                content=context_message
            )
            
            print("📝 Sent initial project context to ChatGPT thread")
            
        except Exception as e:
            print(f"⚠️  Failed to send initial context: {e}")
    
    def create_config_file(self, thread_id: str):
        """Create configuration file for ChatGPT automation"""
        config_sections = [
            ("# ChatGPT Automation Configuration", [
                ("OPENAI_API_KEY", OPENAI_API_KEY),
                ("CHATGPT_THREAD_ID", thread_id),
                ("PROJECT_NAME", PROJECT_NAME),
                ("AUTO_UPLOAD_ENABLED", "true"),
                ("UPLOAD_ENABLED", "true"),
                ("SYNC_DIR", str(SYNC_DIR)),
                ("CHECK_INTERVAL", "10"),
                ("LOG_LEVEL", "INFO")
            ]),
            ("# Webhook Configuration", [
                ("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", "")),
                ("WEBHOOK_SECRET", os.getenv("WEBHOOK_SECRET", "")),
                ("REPO_OWNER", os.getenv("REPO_OWNER", "")),
                ("REPO_NAME", os.getenv("REPO_NAME", "")),
                ("WEBHOOK_URL", os.getenv("WEBHOOK_URL", "")),
                ("PORT", "8001")
            ])
        ]
        
        try:
            with open(CONFIG_FILE, 'w') as f:
                f.write("# ChatGPT Automation Configuration\n")
                f.write(f"# Generated on {datetime.now().isoformat()}\n\n")
                
                for section_title, items in config_sections:
                    f.write(f"{section_title}\n")
                    for key, value in items:
                        f.write(f"{key}={value}\n")
                    f.write("\n")
            
            print(f"✅ Created configuration file: {CONFIG_FILE}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create config file: {e}")
            return False
    
    def verify_setup(self, thread_id: str) -> Dict[str, Any]:
        """Verify the complete setup"""
        verification = {
            "openai_api": False,
            "chatgpt_thread": False,
            "config_file": False,
            "sync_directory": False,
            "github_config": False,
            "webhook_config": False
        }
        
        # Test OpenAI API
        verification["openai_api"] = self.test_openai_connection()
        
        # Test ChatGPT thread
        try:
            thread = self.client.beta.threads.retrieve(thread_id)
            verification["chatgpt_thread"] = bool(thread.id)
        except:
            verification["chatgpt_thread"] = False
        
        # Check config file
        verification["config_file"] = CONFIG_FILE.exists()
        
        # Check sync directory
        SYNC_DIR.mkdir(exist_ok=True)
        verification["sync_directory"] = SYNC_DIR.exists()
        
        # Check GitHub configuration
        github_vars = ["GITHUB_TOKEN", "REPO_OWNER", "REPO_NAME"]
        verification["github_config"] = all(os.getenv(var) for var in github_vars)
        
        # Check webhook configuration
        webhook_vars = ["WEBHOOK_URL", "WEBHOOK_SECRET"]
        verification["webhook_config"] = all(os.getenv(var) for var in webhook_vars)
        
        return verification
    
    def print_setup_summary(self, thread_id: str, verification: Dict[str, Any]):
        """Print setup summary and next steps"""
        print("\n" + "="*60)
        print("🎉 CHATGPT AUTOMATION SETUP COMPLETE!")
        print("="*60)
        
        print(f"📝 ChatGPT Thread ID: {thread_id}")
        print(f"🔗 Thread URL: https://chat.openai.com/chat/{thread_id}")
        print(f"📁 Sync Directory: {SYNC_DIR}")
        print(f"⚙️  Config File: {CONFIG_FILE}")
        
        print("\n📊 Setup Verification:")
        for component, status in verification.items():
            icon = "✅" if status else "❌"
            print(f"   {icon} {component.replace('_', ' ').title()}")
        
        print(f"\n🚀 Next Steps:")
        
        if all(verification.values()):
            print("   1. ✅ All components configured successfully!")
            print("   2. 🔄 Make a commit and push to test automation")
            print("   3. 📱 Check your ChatGPT thread for automatic updates")
            print("   4. 🤖 Start coding - I'll automatically review your changes!")
        else:
            print("   1. ⚠️  Some components need configuration:")
            
            if not verification["github_config"]:
                print("      • Set GITHUB_TOKEN, REPO_OWNER, REPO_NAME")
            if not verification["webhook_config"]:
                print("      • Set WEBHOOK_URL, WEBHOOK_SECRET")
            
            print("   2. 🔧 Complete missing configuration")
            print("   3. 🔄 Re-run setup script to verify")
        
        print(f"\n📚 Usage Commands:")
        print(f"   • Test upload: python chatgpt_api_uploader.py --upload-latest")
        print(f"   • Start service: python auto_upload_service.py")
        print(f"   • Enhanced webhook: python enhanced_webhook_receiver.py")
        
        print(f"\n🎯 Integration Status:")
        print(f"   • GitHub → Webhook: {'✅ Active' if verification['webhook_config'] else '❌ Needs config'}")
        print(f"   • Webhook → Files: {'✅ Active' if verification['sync_directory'] else '❌ Needs setup'}")
        print(f"   • Files → ChatGPT: {'✅ Active' if verification['openai_api'] else '❌ Needs API key'}")

def main():
    """Main setup function"""
    print("🤖 ChatGPT Automation Setup")
    print("=" * 40)
    
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI library not installed")
        print("   Run: pip install openai")
        return 1
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY environment variable is required")
        print("\n📚 Setup Guide:")
        print("   1. Go to: https://platform.openai.com/api-keys")
        print("   2. Create a new API key")
        print("   3. Set environment variable:")
        print("      export OPENAI_API_KEY='your-api-key-here'")
        print("   4. Re-run this setup script")
        return 1
    
    try:
        # Initialize setup
        setup = ChatGPTAutomationSetup()
        
        # Test connection
        if not setup.test_openai_connection():
            return 1
        
        # Create or get thread
        thread_id = setup.create_or_get_thread()
        if not thread_id:
            return 1
        
        # Create config file
        if not setup.create_config_file(thread_id):
            return 1
        
        # Verify setup  
        verification = setup.verify_setup(thread_id)
        
        # Print summary
        setup.print_setup_summary(thread_id, verification)
        
        return 0 if all(verification.values()) else 1
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())