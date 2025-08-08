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


def chunk(
    text: str, 
    max_tokens: int = 400, 
    overlap: int = 60, 
    encode=None
):
    """
    Split text into chunks with deterministic, stable overlap.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
        encode: Optional tokenizer function (e.g., tiktoken.encode)
        
    Yields:
        Text chunks with stable overlap
    """
    if encode is None:
        # Crude character fallback approximation: ~4 chars per token
        step = (max_tokens - overlap) * 4
        size = max_tokens * 4
        for i in range(0, len(text), step):
            yield text[i:i+size]
        return
    
    # Token-aware chunking with proper encoder
    tokens = encode(text)
    step = max_tokens - overlap
    i = 0
    
    while i < len(tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = encode.decode(chunk_tokens) if hasattr(encode, 'decode') else text
        yield chunk_text
        i += step


def smart_chunk_memory(
    content: str,
    max_chunk_size: int = 400,
    overlap: int = 60,
    preserve_sentences: bool = True
) -> list[str]:
    """
    Intelligently chunk memory content preserving sentence boundaries when possible.
    
    Args:
        content: Text content to chunk
        max_chunk_size: Maximum characters per chunk (approximate)
        overlap: Number of characters to overlap
        preserve_sentences: Whether to try to preserve sentence boundaries
        
    Returns:
        List of text chunks
    """
    if len(content) <= max_chunk_size:
        return [content]
    
    chunks = []
    
    if preserve_sentences:
        # Split into sentences first
        import re
        sentences = re.split(r'[.!?]+\s+', content)
        
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max size, save current chunk
            if current_chunk and len(current_chunk + " " + sentence) > max_chunk_size:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                # Add sentence to current chunk
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    else:
        # Simple character-based chunking
        step = max_chunk_size - overlap
        for i in range(0, len(content), step):
            chunk = content[i:i+max_chunk_size]
            chunks.append(chunk)
    
    return chunks


def get_tiktoken_encoder():
    """Get tiktoken encoder for OpenAI models, fallback gracefully"""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    except ImportError:
        return None


def chunk_with_tiktoken(text: str, max_tokens: int = 400, overlap: int = 60):
    """
    Chunk text using tiktoken encoder for accurate token counting
    """
    encoder = get_tiktoken_encoder()
    if encoder is None:
        # Fallback to character-based chunking
        yield from chunk(text, max_tokens, overlap, None)
        return
    
    # Use tiktoken for precise chunking
    yield from chunk(text, max_tokens, overlap, encoder)