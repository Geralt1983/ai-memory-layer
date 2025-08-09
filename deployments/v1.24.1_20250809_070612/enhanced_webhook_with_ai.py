#!/usr/bin/env python3
"""
Enhanced Webhook Receiver with AI Assistant Integration
========================================================

Complete automation: GitHub → Webhook → Memory → AI Assistant
This creates your own AI assistant that automatically learns from every commit.

Features:
- All original webhook functionality  
- Automatic commit processing into AI memory
- Real-time AI assistant updates
- Web interface for chatting with your AI
- Complete GitHub → AI pipeline automation

Usage:
    python enhanced_webhook_with_ai.py

Environment Variables:
    OPENAI_API_KEY          - OpenAI API key for AI assistant
    GITHUB_TOKEN            - GitHub token for webhook
    WEBHOOK_SECRET          - Webhook security
    AI_ASSISTANT_ENABLED    - Enable AI assistant integration (default: true)
    AI_ASSISTANT_PORT       - Port for AI assistant web interface (default: 8002)
"""

import os
import asyncio
from pathlib import Path
from typing import Optional

# Import original webhook components
from github_chatgpt_webhook import *

# Import AI assistant
try:
    from ai_code_assistant import AICodeAssistant
    AI_ASSISTANT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  AI Assistant not available: {e}")
    AI_ASSISTANT_AVAILABLE = False

# Configuration
AI_ASSISTANT_ENABLED = os.getenv("AI_ASSISTANT_ENABLED", "true").lower() == "true"
AI_ASSISTANT_PORT = int(os.getenv("AI_ASSISTANT_PORT", "8002"))

# Enhanced FastAPI app
enhanced_app = FastAPI(
    title="Enhanced GitHub Webhook with AI Assistant",
    version="3.0.0",
    description="Complete automation: GitHub → Webhook → Memory → AI Assistant"
)

# Global AI assistant instance
ai_assistant: Optional[AICodeAssistant] = None

@enhanced_app.on_event("startup")
async def startup_event():
    """Initialize AI assistant on startup"""
    global ai_assistant
    
    print("🚀 Starting Enhanced GitHub Webhook with AI Assistant")
    print(f"📁 Sync directory: {SYNC_DIR}")
    print(f"🔑 GitHub token configured: {bool(GITHUB_TOKEN)}")
    print(f"🔒 Webhook secret configured: {bool(WEBHOOK_SECRET)}")
    print(f"🤖 OpenAI API configured: {bool(OPENAI_API_KEY)}")
    print(f"🧠 AI Assistant enabled: {AI_ASSISTANT_ENABLED}")
    
    if AI_ASSISTANT_AVAILABLE and AI_ASSISTANT_ENABLED and OPENAI_API_KEY:
        try:
            ai_assistant = AICodeAssistant()
            print("✅ AI Assistant initialized and ready")
            
            # Process any existing commit files
            processed = ai_assistant.process_all_commits()
            if processed > 0:
                print(f"📚 Processed {processed} existing commits into AI memory")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize AI assistant: {e}")
            ai_assistant = None

@enhanced_app.get("/")
async def enhanced_root():
    """Enhanced status with AI assistant info"""
    base_status = {
        "service": "Enhanced GitHub Webhook with AI Assistant",
        "status": "running",
        "version": "3.0.0",
        "sync_dir": str(SYNC_DIR),
        "github_configured": bool(GITHUB_TOKEN),
        "webhook_secured": bool(WEBHOOK_SECRET),
        "openai_configured": bool(OPENAI_API_KEY),
        "ai_assistant_enabled": AI_ASSISTANT_ENABLED,
        "ai_assistant_available": AI_ASSISTANT_AVAILABLE,
        "ai_assistant_initialized": ai_assistant is not None
    }
    
    if ai_assistant:
        try:
            ai_stats = ai_assistant.get_stats()
            base_status["ai_assistant_stats"] = ai_stats
        except Exception as e:
            base_status["ai_assistant_error"] = str(e)
    
    return base_status

@enhanced_app.get("/ai-chat")
async def ai_chat_interface():
    """Redirect to AI assistant web interface"""
    if not ai_assistant:
        raise HTTPException(status_code=503, detail="AI Assistant not available")
    
    return {
        "message": "AI Assistant is running",
        "web_interface": f"http://localhost:{AI_ASSISTANT_PORT}",
        "api_endpoints": {
            "chat": "POST /ai/chat",
            "stats": "GET /ai/stats",
            "process_commits": "POST /ai/process-commits"
        }
    }

# AI Assistant API endpoints
class ChatRequest(BaseModel):
    message: str

@enhanced_app.post("/ai/chat")
async def ai_chat_endpoint(request: ChatRequest):
    """Chat with the AI assistant"""
    if not ai_assistant:
        raise HTTPException(status_code=503, detail="AI Assistant not available")
    
    try:
        response = ai_assistant.chat(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI chat error: {e}")

@enhanced_app.get("/ai/stats")
async def ai_stats_endpoint():
    """Get AI assistant statistics"""
    if not ai_assistant:
        raise HTTPException(status_code=503, detail="AI Assistant not available")
    
    return ai_assistant.get_stats()

@enhanced_app.post("/ai/process-commits")
async def ai_process_commits_endpoint():
    """Process all commit files into AI memory"""
    if not ai_assistant:
        raise HTTPException(status_code=503, detail="AI Assistant not available")
    
    count = ai_assistant.process_all_commits()
    return {"processed": count, "message": f"Processed {count} commit files into AI memory"}

# Enhanced webhook processing with AI integration
async def enhanced_sync_to_ai(commit_info: CommitInfo):
    """Enhanced sync that includes AI assistant processing"""
    try:
        # Original file generation
        print(f"🔄 Processing commit {commit_info.sha[:8]} by {commit_info.author}")
        
        # Generate ChatGPT summary
        summary = generate_chatgpt_summary(commit_info)
        
        # Save to file
        filepath = save_sync_file(commit_info, summary)
        
        print(f"✅ Saved sync file: {filepath}")
        
        # Process into AI assistant if available
        if ai_assistant:
            try:
                commit_context = ai_assistant.process_commit_file(filepath)
                print(f"🤖 AI Assistant processed commit {commit_context.sha[:8]}")
                print(f"   Files changed: {len(commit_context.files_changed)}")
                print(f"   Now available for AI chat queries")
            except Exception as e:
                print(f"⚠️  AI processing failed: {e}")
        
        # Print summary for immediate visibility
        print("\n" + "="*60)
        print("COMMIT PROCESSED")
        print("="*60)
        print(summary[:300] + "..." if len(summary) > 300 else summary)
        if ai_assistant:
            print("\n🤖 Ask your AI assistant about this commit!")
            print(f"   Chat at: http://localhost:{AI_ASSISTANT_PORT}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error in enhanced sync: {e}")

# Enhanced webhook endpoint
@enhanced_app.post("/webhook")
async def enhanced_github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Enhanced GitHub webhook with AI assistant integration"""
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
        
        # Enhanced background processing with AI
        background_tasks.add_task(enhanced_sync_to_ai, commit_info)
        
        return {
            "message": "Enhanced webhook received successfully",
            "commit": commit_info.sha[:8],
            "files_changed": len(commit_info.files_changed),
            "author": commit_info.author,
            "ai_assistant_enabled": bool(ai_assistant),
            "ai_chat_available": f"http://localhost:{AI_ASSISTANT_PORT}" if ai_assistant else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Enhanced webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include sync file listing from original webhook
@enhanced_app.get("/sync-files")
async def list_sync_files():
    """List generated sync files"""
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
    
    # Add AI processing status if available
    if ai_assistant:
        try:
            ai_stats = ai_assistant.get_stats()
            return {
                "files": files,
                "ai_stats": ai_stats,
                "ai_chat_url": f"http://localhost:{AI_ASSISTANT_PORT}"
            }
        except:
            pass
    
    return {"files": files}

if __name__ == "__main__":
    print("🔄 Enhanced GitHub Webhook with AI Assistant")
    print(f"📁 Sync directory: {SYNC_DIR}")
    print(f"🔑 GitHub token configured: {bool(GITHUB_TOKEN)}")
    print(f"🔒 Webhook secret configured: {bool(WEBHOOK_SECRET)}")
    print(f"🤖 OpenAI API configured: {bool(OPENAI_API_KEY)}")
    print(f"🧠 AI Assistant enabled: {AI_ASSISTANT_ENABLED}")
    print(f"📡 Webhook listening on port {PORT}")
    print(f"🤖 AI Chat available on port {AI_ASSISTANT_PORT}")
    
    # Create sync directory
    SYNC_DIR.mkdir(exist_ok=True)
    
    # Run the enhanced webhook server
    uvicorn.run(enhanced_app, host="0.0.0.0", port=PORT)