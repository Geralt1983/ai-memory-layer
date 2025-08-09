#!/usr/bin/env python3
"""
Automated ChatGPT Upload Service
=================================

Background service that automatically uploads GitHub webhook sync files to ChatGPT.
This integrates with the webhook receiver to create a fully automated pipeline:
GitHub Push → Webhook → File Generation → ChatGPT Upload

Features:
- Runs as background service alongside webhook receiver
- Watches for new .md files in chatgpt-sync directory
- Automatically uploads to ChatGPT via OpenAI API
- Maintains conversation continuity with thread management
- Handles rate limiting and error recovery
- Comprehensive logging and monitoring

Usage:
    python auto_upload_service.py

Integration with Webhook:
- Webhook creates .md files in chatgpt-sync/
- This service detects new files and uploads them
- Creates seamless GitHub → ChatGPT automation

Environment Variables:
    OPENAI_API_KEY          - OpenAI API key (required)
    CHATGPT_THREAD_ID       - Persistent thread ID for conversations
    UPLOAD_ENABLED          - Enable/disable auto uploads (default: true)
    SYNC_DIR                - Directory to monitor (default: ./chatgpt-sync)
    CHECK_INTERVAL          - File check interval in seconds (default: 10)
"""

import os
import time
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

# Import the uploader class
from chatgpt_api_uploader import ChatGPTUploader

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHATGPT_THREAD_ID = os.getenv("CHATGPT_THREAD_ID", "")
UPLOAD_ENABLED = os.getenv("UPLOAD_ENABLED", "true").lower() == "true"
SYNC_DIR = Path(os.getenv("SYNC_DIR", "./chatgpt-sync"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SYNC_DIR / 'auto_upload.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoUploadService")

class AutoUploadService:
    """Background service for automatic ChatGPT uploads"""
    
    def __init__(self):
        self.uploader = ChatGPTUploader()
        self.running = False
        self.known_files = set()
        self.thread = None
        
        # Initialize known files
        if SYNC_DIR.exists():
            self.known_files = {f.name for f in SYNC_DIR.glob("*.md")}
            logger.info(f"Initialized with {len(self.known_files)} existing files")
    
    def start(self):
        """Start the background monitoring service"""
        if self.running:
            logger.warning("Service is already running")
            return
        
        logger.info("🤖 Starting ChatGPT Auto-Upload Service")
        logger.info(f"📁 Monitoring directory: {SYNC_DIR}")
        logger.info(f"⏱️  Check interval: {CHECK_INTERVAL} seconds")
        logger.info(f"🔄 Upload enabled: {UPLOAD_ENABLED}")
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        logger.info("✅ Auto-upload service started")
    
    def stop(self):
        """Stop the background monitoring service"""
        if not self.running:
            return
        
        logger.info("🛑 Stopping auto-upload service...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("✅ Auto-upload service stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        logger.info("👀 File monitoring started")
        
        while self.running:
            try:
                self._check_for_new_files()
                time.sleep(CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(CHECK_INTERVAL)
        
        logger.info("👋 File monitoring stopped")
    
    def _check_for_new_files(self):
        """Check for new .md files and upload them"""
        if not SYNC_DIR.exists():
            return
        
        current_files = {f.name for f in SYNC_DIR.glob("*.md")}
        new_files = current_files - self.known_files
        
        if new_files:
            logger.info(f"🆕 Detected {len(new_files)} new file(s): {', '.join(new_files)}")
            
            for new_file in sorted(new_files):
                self._process_new_file(SYNC_DIR / new_file)
        
        self.known_files = current_files
    
    def _process_new_file(self, filepath: Path):
        """Process a newly detected file"""
        try:
            logger.info(f"📤 Processing new file: {filepath.name}")
            
            if not UPLOAD_ENABLED:
                logger.info(f"⏸️  Upload disabled, skipping {filepath.name}")
                return
            
            # Wait briefly to ensure file is fully written
            time.sleep(2)
            
            # Upload to ChatGPT
            success = self.uploader.upload_sync_file(filepath)
            
            if success:
                logger.info(f"✅ Successfully uploaded {filepath.name} to ChatGPT")
                
                # Log success statistics
                self._log_upload_stats()
            else:
                logger.error(f"❌ Failed to upload {filepath.name}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {filepath.name}: {e}")
    
    def _log_upload_stats(self):
        """Log upload statistics"""
        try:
            total_uploads = len(self.uploader.upload_log)
            recent_uploads = sum(1 for record in self.uploader.upload_log.values() 
                               if (datetime.now() - datetime.fromisoformat(record.uploaded_at)).days < 1)
            
            logger.info(f"📊 Upload stats: {total_uploads} total, {recent_uploads} in last 24h")
            
        except Exception as e:
            logger.debug(f"Failed to log stats: {e}")
    
    def get_status(self) -> dict:
        """Get service status information"""
        return {
            "running": self.running,
            "sync_dir": str(SYNC_DIR),
            "upload_enabled": UPLOAD_ENABLED,
            "check_interval": CHECK_INTERVAL,
            "known_files": len(self.known_files),
            "total_uploads": len(self.uploader.upload_log),
            "openai_configured": bool(OPENAI_API_KEY),
            "thread_id": CHATGPT_THREAD_ID or "auto-generated"
        }

# Global service instance
_service_instance = None

def get_service() -> AutoUploadService:
    """Get or create the global service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = AutoUploadService()
    return _service_instance

def start_service():
    """Start the auto-upload service"""
    service = get_service()
    service.start()
    return service

def stop_service():
    """Stop the auto-upload service"""
    global _service_instance
    if _service_instance:
        _service_instance.stop()
        _service_instance = None

def main():
    """Main entry point for running as standalone service"""
    import signal
    import sys
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY environment variable is required")
        print("\nSet it with:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    # Ensure sync directory exists
    SYNC_DIR.mkdir(exist_ok=True)
    
    # Start service
    service = start_service()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"📡 Received signal {signum}, shutting down...")
        stop_service()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep main thread alive
        while service.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    finally:
        stop_service()
    
    return 0

if __name__ == "__main__":
    exit(main())