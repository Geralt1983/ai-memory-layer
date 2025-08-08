#!/usr/bin/env python3
"""
Continuous Import Monitor
Real-time monitoring of the ChatGPT import progress
"""

import json
import os
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_process_status():
    """Check if the import process is running"""
    try:
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'import_conversations_optimized' in result.stdout
    except:
        return False

def format_time_remaining(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_import_rate(progress_file):
    """Calculate import rate based on recent progress"""
    if not os.path.exists(progress_file):
        return 0
    
    # Read current progress
    with open(progress_file, 'r') as f:
        current = json.load(f)
    
    # Store previous reading for rate calculation
    rate_file = "./data/rate_tracking.json"
    current_time = time.time()
    current_count = current.get("processed_count", 0)
    
    if os.path.exists(rate_file):
        with open(rate_file, 'r') as f:
            previous = json.load(f)
        
        time_diff = current_time - previous.get("timestamp", current_time)
        count_diff = current_count - previous.get("count", current_count)
        
        if time_diff > 0:
            rate = count_diff / time_diff
        else:
            rate = 0
    else:
        rate = 0
    
    # Save current reading
    os.makedirs("./data", exist_ok=True)
    with open(rate_file, 'w') as f:
        json.dump({"timestamp": current_time, "count": current_count}, f)
    
    return rate

def monitor_import():
    """Continuously monitor the import progress"""
    progress_file = "./data/import_progress.json"
    memories_file = "./data/chatgpt_memories.json"
    log_file = "./import.log"
    
    print("🚀 Starting Continuous Import Monitor")
    print("Press Ctrl+C to exit")
    time.sleep(2)
    
    last_update_time = None
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("=" * 80)
            print("📊 ChatGPT IMPORT MONITOR - Live Status")
            print("=" * 80)
            print(f"🕐 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Process status
            is_running = get_process_status()
            status_icon = "🟢" if is_running else "🔴"
            status_text = "RUNNING" if is_running else "STOPPED"
            print(f"{status_icon} Process Status: {status_text}")
            print()
            
            # Progress information
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                
                processed = progress.get("processed_count", 0)
                total = progress.get("total_count", 0)
                errors = progress.get("errors", 0)
                completion = progress.get("completion_percentage", 0)
                timestamp = progress.get("timestamp")
                
                # Calculate progress bar
                bar_width = 50
                filled = int(bar_width * completion / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                print(f"📈 Progress: {processed:,} / {total:,} messages")
                print(f"[{bar}] {completion:.1f}%")
                print()
                
                # Import rate and ETA
                rate = get_import_rate(progress_file)
                if rate > 0 and processed < total:
                    remaining = total - processed
                    eta_seconds = remaining / rate
                    eta_str = format_time_remaining(eta_seconds)
                    print(f"⚡ Rate: {rate:.1f} messages/second")
                    print(f"⏰ ETA: {eta_str}")
                else:
                    print(f"⚡ Rate: Calculating...")
                    print(f"⏰ ETA: Calculating...")
                print()
                
                # Error information
                error_icon = "✅" if errors == 0 else "⚠️"
                print(f"{error_icon} Errors: {errors}")
                print()
                
                # Last update info
                if timestamp:
                    try:
                        last_update = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_since = datetime.now() - last_update.replace(tzinfo=None)
                        
                        if time_since.total_seconds() < 60:
                            update_status = f"Just now ({time_since.total_seconds():.0f}s ago)"
                            update_icon = "🟢"
                        elif time_since.total_seconds() < 300:  # 5 minutes
                            update_status = f"{time_since.total_seconds()/60:.1f}m ago"
                            update_icon = "🟡"
                        else:
                            update_status = f"{time_since.total_seconds()/60:.0f}m ago (may be stuck)"
                            update_icon = "🔴"
                        
                        print(f"{update_icon} Last Update: {update_status}")
                    except:
                        print(f"🟡 Last Update: {timestamp}")
                print()
                
                # Completion status
                if completion >= 100:
                    print("🎉" * 20)
                    print("🎉 IMPORT COMPLETE! 🎉")
                    print("🎉" * 20)
                    print()
                    print("🚀 Your ChatGPT conversations are now fully searchable!")
                    break
                
            else:
                print("📂 No progress file found - import may not have started")
                print()
            
            # Memory file status
            if os.path.exists(memories_file):
                try:
                    file_size = os.path.getsize(memories_file) / (1024 * 1024)  # MB
                    print(f"💾 Memory File: {file_size:.1f} MB")
                except:
                    print(f"💾 Memory File: Present")
            else:
                print(f"💾 Memory File: Not created yet")
            print()
            
            # Recent log entries
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Get last few embedding operations
                    recent_embeddings = [line for line in lines[-50:] if 'embed_text' in line and 'text_length' in line]
                    if recent_embeddings:
                        print("📝 Recent Activity:")
                        for line in recent_embeddings[-3:]:
                            try:
                                log_data = json.loads(line.strip())
                                text_len = log_data.get('text_length', 0)
                                duration = log_data.get('duration_ms', 0)
                                timestamp = log_data.get('timestamp', '').split('T')[1][:8]
                                print(f"  {timestamp} - Embedded {text_len} chars in {duration:.0f}ms")
                            except:
                                pass
                    else:
                        print("📝 Recent Activity: No recent embedding operations")
                except:
                    print("📝 Recent Activity: Unable to read log")
            else:
                print("📝 Log File: Not found")
            
            print()
            print("=" * 80)
            print("💡 Commands: python3 check_import_status.py | tail -f import.log")
            print("⏹️  To stop: kill the import process or Ctrl+C here")
            print("=" * 80)
            
            # Wait before next update
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")
        print("💡 Import process continues running in background")
        print("💡 Run this script again anytime to resume monitoring")

if __name__ == "__main__":
    monitor_import()