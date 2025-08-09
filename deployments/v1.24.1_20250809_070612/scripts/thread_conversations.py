#!/usr/bin/env python3
"""
Create conversation threads from cleaned memories to improve context quality
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

def thread_memories(input_file: str, output_file: str, time_gap_minutes: int = 30):
    """Thread memories into conversation blocks"""
    
    print(f"🧵 Threading memories from {input_file}")
    print(f"📄 Output will be saved to {output_file}")
    print(f"⏱️ Using {time_gap_minutes} minute conversation gaps")
    
    # Load memories
    with open(input_file, 'r', encoding='utf-8') as f:
        memories = json.load(f)
    
    print(f"📊 Loaded {len(memories):,} memories")
    
    # Sort by timestamp
    def get_timestamp(memory):
        ts = memory.get('timestamp', memory.get('created_at', ''))
        if not ts:
            return datetime.min
        try:
            if 'T' in ts:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(ts)
        except:
            return datetime.min
    
    sorted_memories = sorted(memories, key=get_timestamp)
    print("✅ Memories sorted by timestamp")
    
    # Group into threads
    threads = []
    current_thread = []
    last_timestamp = None
    time_gap = timedelta(minutes=time_gap_minutes)
    
    for memory in sorted_memories:
        current_timestamp = get_timestamp(memory)
        
        if last_timestamp and current_timestamp:
            time_diff = abs(current_timestamp - last_timestamp)
            
            if time_diff > time_gap:
                # Save current thread and start new one
                if current_thread:
                    threads.append(current_thread)
                current_thread = [memory]
            else:
                # Continue current thread
                current_thread.append(memory)
        else:
            # First memory or no timestamp
            current_thread.append(memory)
        
        last_timestamp = current_timestamp
    
    # Add final thread
    if current_thread:
        threads.append(current_thread)
    
    print(f"🧵 Created {len(threads):,} conversation threads")
    
    # Merge threads into conversation blocks
    conversation_blocks = []
    total_merged = 0
    
    for thread in threads:
        if len(thread) == 1:
            # Single memory - keep as is if substantial
            memory = thread[0]
            if len(memory.get('content', '')) > 25:
                conversation_blocks.append(memory)
        else:
            # Multiple memories - merge them
            content_pieces = []
            for memory in thread:
                content = memory.get('content', '').strip()
                if content and len(content) > 5:  # Skip very short pieces
                    content_pieces.append(content)
            
            if content_pieces:
                # Create conversation block
                merged_content = " | ".join(content_pieces)
                
                if len(merged_content) > 30:  # Only keep meaningful blocks
                    conversation_block = {
                        'content': merged_content,
                        'timestamp': thread[0].get('timestamp', thread[0].get('created_at', '')),
                        'metadata': {
                            'conversation_thread': True,
                            'original_count': len(thread),
                            'content_pieces': len(content_pieces),
                            'thread_length': len(merged_content),
                            'conversation_type': detect_conversation_type(merged_content)
                        }
                    }
                    conversation_blocks.append(conversation_block)
                    total_merged += len(thread)
    
    # Save threaded conversations
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_blocks, f, indent=2, ensure_ascii=False)
    
    # Calculate stats
    original_count = len(memories)
    final_count = len(conversation_blocks)
    reduction = ((original_count - final_count) / original_count) * 100 if original_count > 0 else 0
    
    # Analyze conversation types
    types = {}
    total_length = 0
    for block in conversation_blocks:
        conv_type = block.get('metadata', {}).get('conversation_type', 'unknown')
        types[conv_type] = types.get(conv_type, 0) + 1
        total_length += len(block.get('content', ''))
    
    avg_length = total_length // final_count if final_count > 0 else 0
    
    print(f"\n📊 Threading Results:")
    print(f"   Original memories: {original_count:,}")
    print(f"   Conversation blocks: {final_count:,}")
    print(f"   Reduction: {reduction:.1f}%")
    print(f"   Average block length: {avg_length} chars")
    print(f"   Memories merged: {total_merged:,}")
    
    print(f"\n🏷️ Conversation types:")
    for conv_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {conv_type}: {count:,}")
    
    return {
        'original_count': original_count,
        'final_count': final_count,
        'reduction_percent': reduction,
        'avg_length': avg_length,
        'types': types
    }

def detect_conversation_type(content: str) -> str:
    """Detect type of conversation from content"""
    content_lower = content.lower()
    
    # Question & Answer patterns
    if '?' in content and any(word in content_lower for word in ['answer', 'because', 'solution', 'yes', 'no']):
        return 'qa_discussion'
    
    # Code-related
    elif any(word in content_lower for word in ['code', 'function', 'python', 'script', 'import', 'def', 'class']):
        return 'programming'
    
    # Problem solving
    elif any(word in content_lower for word in ['error', 'problem', 'fix', 'issue', 'troubleshoot', 'debug']):
        return 'troubleshooting'
    
    # Learning/tutorial
    elif any(word in content_lower for word in ['learn', 'tutorial', 'how to', 'explain', 'understand']):
        return 'learning'
    
    # Task/project
    elif any(word in content_lower for word in ['project', 'task', 'build', 'create', 'make', 'develop']):
        return 'project_work'
    
    # Personal/emotional
    elif any(word in content_lower for word in ['feel', 'tired', 'stressed', 'happy', 'sad', 'angry']):
        return 'personal_emotional'
    
    # Default
    else:
        return 'general_conversation'

if __name__ == "__main__":
    # Check if cleaned memories exist
    cleaned_file = "data/chatgpt_memories_cleaned.json"
    threaded_file = "data/chatgpt_conversations_threaded.json"
    
    if not Path(cleaned_file).exists():
        print(f"❌ Cleaned memories file not found: {cleaned_file}")
        print("   Please run the memory cleaning process first")
        sys.exit(1)
    
    # Thread the conversations
    stats = thread_memories(cleaned_file, threaded_file, time_gap_minutes=30)
    
    print(f"\n✅ Conversation threading complete!")
    print(f"📄 Threaded conversations saved to: {threaded_file}")
    print(f"🎯 Ready for improved contextual search!")