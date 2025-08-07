"""
Memory Chunking and Fragment Merging Utilities
"""
from typing import List
from datetime import datetime, timedelta
from core.memory import Memory

def merge_conversation_fragments(memories: List[Memory], 
                                time_window_minutes: int = 30,
                                min_fragment_length: int = 40) -> List[Memory]:
    """
    Merge short memory fragments into coherent conversation chunks
    
    Args:
        memories: List of Memory objects to process
        time_window_minutes: Time window to consider memories as part of same conversation
        min_fragment_length: Minimum length to consider as standalone memory
    
    Returns:
        List of merged Memory objects with better context
    """
    if not memories:
        return []
    
    # Sort memories by timestamp
    sorted_memories = sorted(memories, key=lambda x: x.timestamp)
    
    merged = []
    current_chunk = []
    current_chunk_start = None
    
    for memory in sorted_memories:
        # Skip empty memories
        if not memory.content or len(memory.content.strip()) == 0:
            continue
            
        # If this is a substantial memory, save current chunk and add this as standalone
        if len(memory.content) >= min_fragment_length * 3:  # 120+ chars is substantial
            if current_chunk:
                # Save accumulated chunk
                merged_content = " ".join([m.content for m in current_chunk])
                if len(merged_content) > 15:  # Only save meaningful chunks
                    merged.append(Memory(
                        content=merged_content,
                        timestamp=current_chunk_start or current_chunk[0].timestamp,
                        metadata={"merged": True, "fragment_count": len(current_chunk)}
                    ))
                current_chunk = []
                current_chunk_start = None
            
            # Add substantial memory as-is
            merged.append(memory)
            
        # If this is a fragment
        elif len(memory.content) < min_fragment_length:
            # Check if it belongs to current chunk (within time window)
            if current_chunk and current_chunk_start:
                time_diff = memory.timestamp - current_chunk[-1].timestamp
                if time_diff < timedelta(minutes=time_window_minutes):
                    current_chunk.append(memory)
                else:
                    # Save current chunk and start new one
                    merged_content = " ".join([m.content for m in current_chunk])
                    if len(merged_content) > 15:
                        merged.append(Memory(
                            content=merged_content,
                            timestamp=current_chunk_start,
                            metadata={"merged": True, "fragment_count": len(current_chunk)}
                        ))
                    current_chunk = [memory]
                    current_chunk_start = memory.timestamp
            else:
                # Start new chunk
                current_chunk = [memory]
                current_chunk_start = memory.timestamp
                
        # Medium-sized memory
        else:
            if current_chunk:
                # Save accumulated chunk
                merged_content = " ".join([m.content for m in current_chunk])
                if len(merged_content) > 15:
                    merged.append(Memory(
                        content=merged_content,
                        timestamp=current_chunk_start or current_chunk[0].timestamp,
                        metadata={"merged": True, "fragment_count": len(current_chunk)}
                    ))
                current_chunk = []
                current_chunk_start = None
            
            # Add medium memory as-is
            merged.append(memory)
    
    # Don't forget last chunk
    if current_chunk:
        merged_content = " ".join([m.content for m in current_chunk])
        if len(merged_content) > 15:
            merged.append(Memory(
                content=merged_content,
                timestamp=current_chunk_start or current_chunk[0].timestamp,
                metadata={"merged": True, "fragment_count": len(current_chunk)}
            ))
    
    return merged

def group_by_conversation(memories: List[Memory], 
                         time_gap_minutes: int = 60) -> List[List[Memory]]:
    """
    Group memories into conversation threads based on time gaps
    
    Args:
        memories: List of Memory objects
        time_gap_minutes: Minutes of inactivity to consider new conversation
    
    Returns:
        List of conversation groups (each group is a list of memories)
    """
    if not memories:
        return []
    
    sorted_memories = sorted(memories, key=lambda x: x.timestamp)
    conversations = []
    current_conversation = [sorted_memories[0]]
    
    for i in range(1, len(sorted_memories)):
        time_diff = sorted_memories[i].timestamp - sorted_memories[i-1].timestamp
        
        if time_diff > timedelta(minutes=time_gap_minutes):
            # Start new conversation
            conversations.append(current_conversation)
            current_conversation = [sorted_memories[i]]
        else:
            # Continue current conversation
            current_conversation.append(sorted_memories[i])
    
    # Add last conversation
    if current_conversation:
        conversations.append(current_conversation)
    
    return conversations

def extract_qa_pairs(memories: List[Memory]) -> List[dict]:
    """
    Extract question-answer pairs from memory list
    
    Args:
        memories: List of Memory objects
    
    Returns:
        List of Q&A pair dictionaries
    """
    qa_pairs = []
    
    for i in range(len(memories) - 1):
        current = memories[i].content
        next_mem = memories[i + 1].content
        
        # Simple heuristic: if current ends with ? and next is longer response
        if current.rstrip().endswith('?') and len(next_mem) > len(current) * 1.5:
            qa_pairs.append({
                "question": current,
                "answer": next_mem,
                "q_timestamp": memories[i].timestamp,
                "a_timestamp": memories[i + 1].timestamp
            })
    
    return qa_pairs