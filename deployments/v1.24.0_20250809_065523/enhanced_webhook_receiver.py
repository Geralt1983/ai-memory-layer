#!/usr/bin/env python3
"""
Enhanced GitHub Webhook Receiver with ChatGPT Auto-Upload
===========================================================

Enhanced version of the webhook receiver that integrates with ChatGPT auto-upload.
This creates a complete automated pipeline: GitHub → Webhook → File Generation → ChatGPT

Features:
- All original webhook receiver functionality
- Integrated ChatGPT auto-upload service
- Real-time status monitoring
- Comprehensive API endpoints for monitoring
- Background upload service management

Usage:
    python enhanced_webhook_receiver.py

Environment Variables:
    All original webhook variables plus:
    OPENAI_API_KEY          - OpenAI API key for ChatGPT uploads
    CHATGPT_THREAD_ID       - ChatGPT thread ID for conversations
    AUTO_UPLOAD_ENABLED     - Enable automatic ChatGPT uploads (default: true)
"""

# Import original webhook receiver code
import sys
import os
from pathlib import Path

# Import the auto-upload service
try:
    from auto_upload_service import get_service, start_service, stop_service
    UPLOAD_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Auto-upload service not available: {e}")
    UPLOAD_SERVICE_AVAILABLE = False

# Import original webhook components
from github_chatgpt_webhook import *

# Additional configuration for auto-upload
AUTO_UPLOAD_ENABLED = os.getenv("AUTO_UPLOAD_ENABLED", "true").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Enhanced FastAPI app with upload integration
enhanced_app = FastAPI(
    title="Enhanced GitHub → ChatGPT Webhook Sync", 
    version="2.0.0",
    description="Complete automation: GitHub webhooks → File generation → ChatGPT upload"
)

# Include all original routes
enhanced_app.include_router(app.router)

@enhanced_app.on_event("startup")
async def startup_event():
    """Start auto-upload service on application startup"""
    print("🚀 Starting Enhanced GitHub → ChatGPT Webhook Receiver")
    print(f"📁 Sync directory: {SYNC_DIR}")
    print(f"🔑 GitHub token configured: {bool(GITHUB_TOKEN)}")
    print(f"🔒 Webhook secret configured: {bool(WEBHOOK_SECRET)}")
    print(f"🤖 OpenAI API configured: {bool(OPENAI_API_KEY)}")
    print(f"🔄 Auto-upload enabled: {AUTO_UPLOAD_ENABLED}")
    
    if UPLOAD_SERVICE_AVAILABLE and AUTO_UPLOAD_ENABLED and OPENAI_API_KEY:
        try:
            start_service()
            print("✅ ChatGPT auto-upload service started")
        except Exception as e:
            print(f"⚠️  Failed to start auto-upload service: {e}")

@enhanced_app.on_event("shutdown")
async def shutdown_event():
    """Stop auto-upload service on application shutdown"""
    print("🛑 Shutting down enhanced webhook receiver...")
    
    if UPLOAD_SERVICE_AVAILABLE:
        try:
            stop_service()
            print("✅ ChatGPT auto-upload service stopped")
        except Exception as e:
            print(f"⚠️  Error stopping auto-upload service: {e}")

@enhanced_app.get("/")
async def enhanced_root():
    """Enhanced status endpoint with upload service info"""
    base_status = {
        "service": "Enhanced GitHub → ChatGPT Webhook Sync",
        "status": "running",
        "version": "2.0.0",
        "sync_dir": str(SYNC_DIR),
        "github_configured": bool(GITHUB_TOKEN),
        "webhook_secured": bool(WEBHOOK_SECRET),
        "openai_configured": bool(OPENAI_API_KEY),
        "auto_upload_enabled": AUTO_UPLOAD_ENABLED,
        "upload_service_available": UPLOAD_SERVICE_AVAILABLE
    }
    
    # Add upload service status if available
    if UPLOAD_SERVICE_AVAILABLE:
        try:
            service = get_service()
            upload_status = service.get_status()
            base_status.update({
                "upload_service_status": upload_status
            })
        except Exception as e:
            base_status["upload_service_error"] = str(e)
    
    return base_status

@enhanced_app.get("/upload-status")
async def upload_status():
    """Get detailed upload service status"""
    if not UPLOAD_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Upload service not available")
    
    try:
        service = get_service()
        return {
            "status": "success",
            "upload_service": service.get_status(),
            "recent_uploads": list(service.uploader.upload_log.values())[-10:]  # Last 10 uploads
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get upload status: {e}")

@enhanced_app.post("/upload-control")
async def upload_control(action: str):
    """Control upload service (start/stop/restart)"""
    if not UPLOAD_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Upload service not available")
    
    try:
        if action == "start":
            start_service()
            return {"status": "success", "message": "Upload service started"}
        elif action == "stop":
            stop_service()
            return {"status": "success", "message": "Upload service stopped"}
        elif action == "restart":
            stop_service()
            start_service()
            return {"status": "success", "message": "Upload service restarted"}
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use: start, stop, restart")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to {action} upload service: {e}")

@enhanced_app.post("/upload-latest")
async def upload_latest():
    """Manually upload the latest sync file to ChatGPT"""
    if not UPLOAD_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Upload service not available")
    
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")
    
    try:
        service = get_service()
        success = service.uploader.upload_latest_file()
        
        if success:
            return {"status": "success", "message": "Latest file uploaded to ChatGPT"}
        else:
            raise HTTPException(status_code=500, detail="Failed to upload latest file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@enhanced_app.get("/chatgpt-integration")
async def chatgpt_integration_info():
    """Get ChatGPT integration information and setup guide"""
    return {
        "integration_status": {
            "openai_api_configured": bool(OPENAI_API_KEY),
            "auto_upload_enabled": AUTO_UPLOAD_ENABLED,
            "upload_service_available": UPLOAD_SERVICE_AVAILABLE
        },
        "setup_guide": {
            "step_1": "Get OpenAI API key from https://platform.openai.com/api-keys",
            "step_2": "Set OPENAI_API_KEY environment variable",
            "step_3": "Optionally set CHATGPT_THREAD_ID for conversation continuity",
            "step_4": "Enable AUTO_UPLOAD_ENABLED=true for automatic uploads",
            "step_5": "Make commits to trigger automatic ChatGPT integration"
        },
        "api_endpoints": {
            "upload_latest": "POST /upload-latest - Manual upload latest file",
            "upload_status": "GET /upload-status - Check upload service status", 
            "upload_control": "POST /upload-control - Control upload service",
            "integration_info": "GET /chatgpt-integration - This endpoint"
        }
    }

# Enhanced background sync function
async def enhanced_sync_to_chatgpt(commit_info: CommitInfo):
    """Enhanced sync that includes auto-upload trigger"""
    try:
        # Original file generation
        await sync_to_chatgpt(commit_info)
        
        # Trigger auto-upload if enabled
        if UPLOAD_SERVICE_AVAILABLE and AUTO_UPLOAD_ENABLED and OPENAI_API_KEY:
            print("🤖 Auto-upload service will process the new file")
        elif not AUTO_UPLOAD_ENABLED:
            print("ℹ️  Auto-upload disabled. Use /upload-latest to upload manually")
        elif not OPENAI_API_KEY:
            print("⚠️  OpenAI API key not configured. Set OPENAI_API_KEY to enable uploads")
        
    except Exception as e:
        print(f"❌ Error in enhanced sync: {e}")

# Replace the webhook endpoint with enhanced version
@enhanced_app.post("/webhook")
async def enhanced_github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Enhanced GitHub webhook handler with auto-upload integration"""
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
        
        # Enhanced background task
        background_tasks.add_task(enhanced_sync_to_chatgpt, commit_info)
        
        return {
            "message": "Enhanced webhook received successfully",
            "commit": commit_info.sha[:8],
            "files_changed": len(commit_info.files_changed),
            "author": commit_info.author,
            "auto_upload_enabled": AUTO_UPLOAD_ENABLED,
            "chatgpt_integration": bool(OPENAI_API_KEY)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Enhanced webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🔄 Enhanced GitHub → ChatGPT Webhook Sync Server")
    print(f"📁 Sync directory: {SYNC_DIR}")
    print(f"🔑 GitHub token configured: {bool(GITHUB_TOKEN)}")
    print(f"🔒 Webhook secret configured: {bool(WEBHOOK_SECRET)}")
    print(f"🤖 OpenAI API configured: {bool(OPENAI_API_KEY)}")
    print(f"🔄 Auto-upload enabled: {AUTO_UPLOAD_ENABLED}")
    print(f"📡 Listening on port {PORT}")
    
    # Create sync directory
    SYNC_DIR.mkdir(exist_ok=True)
    
    # Use enhanced app
    uvicorn.run(enhanced_app, host="0.0.0.0", port=PORT)