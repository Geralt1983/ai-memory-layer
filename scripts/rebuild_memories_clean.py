#!/usr/bin/env python3
"""
Memory Rebuilder v1.2.1 - Clean and reindex ChatGPT memories
Merges fragments, groups conversations, and rebuilds FAISS index
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Use the existing Memory class from the codebase
from datetime import datetime

def load_raw_memories(filepath: str = "data/chatgpt_memories.json") -> List[Dict]:
    """Load raw memory data from JSON file"""
    print(f"📂 Loading raw memories from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        memories = json.load(f)
    print(f"✅ Loaded {len(memories):,} raw memories")
    return memories

def analyze_memory_quality(memories: List[Dict]) -> Dict[str, Any]:
    """Analyze memory quality distribution"""
    stats = {
        'total': len(memories),
        'empty': 0,
        'tiny': 0,      # < 20 chars
        'small': 0,     # 20-40 chars
        'medium': 0,    # 40-200 chars
        'large': 0,     # 200-1000 chars
        'huge': 0,      # > 1000 chars
        'fragments_to_merge': []
    }
    
    for i, mem in enumerate(memories):
        content = mem.get('content', '')
        length = len(content)
        
        if length == 0:
            stats['empty'] += 1
        elif length < 20:
            stats['tiny'] += 1
            stats['fragments_to_merge'].append(i)
        elif length < 40:
            stats['small'] += 1
            stats['fragments_to_merge'].append(i)
        elif length < 200:
            stats['medium'] += 1
        elif length < 1000:
            stats['large'] += 1
        else:
            stats['huge'] += 1
    
    return stats

def group_memories_by_time(memories: List[Dict], gap_minutes: int = 30) -> List[List[Dict]]:
    """Group memories into conversations based on timestamp proximity"""
    if not memories:
        return []
    
    # Sort by timestamp
    sorted_mems = sorted(memories, key=lambda x: x.get('timestamp', ''))
    
    conversations = []
    current_conv = [sorted_mems[0]]
    
    for i in range(1, len(sorted_mems)):
        try:
            # Parse timestamps
            curr_time = datetime.fromisoformat(sorted_mems[i].get('timestamp', '').replace('Z', '+00:00'))
            prev_time = datetime.fromisoformat(sorted_mems[i-1].get('timestamp', '').replace('Z', '+00:00'))
            
            # Check time gap
            if (curr_time - prev_time) > timedelta(minutes=gap_minutes):
                # Start new conversation
                if current_conv:
                    conversations.append(current_conv)
                current_conv = [sorted_mems[i]]
            else:
                # Continue current conversation
                current_conv.append(sorted_mems[i])
        except:
            # If timestamp parsing fails, continue current conversation
            current_conv.append(sorted_mems[i])
    
    # Add last conversation
    if current_conv:
        conversations.append(current_conv)
    
    return conversations

def merge_conversation_fragments(conversation: List[Dict], min_length: int = 40) -> List[Dict]:
    """Merge small fragments within a conversation into coherent segments"""
    if not conversation:
        return []
    
    merged = []
    fragment_buffer = []
    
    for mem in conversation:
        content = mem.get('content', '').strip()
        
        if not content:
            continue
            
        # If this is a substantial piece, save buffer and add this
        if len(content) >= min_length * 2:  # 80+ chars is substantial
            if fragment_buffer:
                # Merge buffered fragments
                merged_content = " ".join([m.get('content', '') for m in fragment_buffer])
                if len(merged_content) > 15:
                    merged.append({
                        'content': merged_content,
                        'timestamp': fragment_buffer[0].get('timestamp'),
                        'metadata': {
                            **fragment_buffer[0].get('metadata', {}),
                            'merged': True,
                            'fragment_count': len(fragment_buffer)
                        }
                    })
                fragment_buffer = []
            
            # Add substantial memory as-is
            merged.append(mem)
            
        # If it's a fragment, buffer it
        elif len(content) < min_length:
            fragment_buffer.append(mem)
            
        # Medium-sized, check if it's a question
        elif content.rstrip().endswith('?'):
            # Questions often pair with answers, keep separate
            if fragment_buffer:
                merged_content = " ".join([m.get('content', '') for m in fragment_buffer])
                if len(merged_content) > 15:
                    merged.append({
                        'content': merged_content,
                        'timestamp': fragment_buffer[0].get('timestamp'),
                        'metadata': {
                            **fragment_buffer[0].get('metadata', {}),
                            'merged': True,
                            'fragment_count': len(fragment_buffer)
                        }
                    })
                fragment_buffer = []
            merged.append(mem)
        else:
            # Medium-sized non-question, add to buffer or save
            if fragment_buffer and len(" ".join([m.get('content', '') for m in fragment_buffer])) < 100:
                fragment_buffer.append(mem)
            else:
                if fragment_buffer:
                    merged_content = " ".join([m.get('content', '') for m in fragment_buffer])
                    if len(merged_content) > 15:
                        merged.append({
                            'content': merged_content,
                            'timestamp': fragment_buffer[0].get('timestamp'),
                            'metadata': {
                                **fragment_buffer[0].get('metadata', {}),
                                'merged': True,
                                'fragment_count': len(fragment_buffer)
                            }
                        })
                    fragment_buffer = []
                merged.append(mem)
    
    # Don't forget the last buffer
    if fragment_buffer:
        merged_content = " ".join([m.get('content', '') for m in fragment_buffer])
        if len(merged_content) > 15:
            merged.append({
                'content': merged_content,
                'timestamp': fragment_buffer[0].get('timestamp'),
                'metadata': {
                    **fragment_buffer[0].get('metadata', {}),
                    'merged': True,
                    'fragment_count': len(fragment_buffer)
                }
            })
    
    return merged

def preserve_qa_pairs(conversation: List[Dict]) -> List[Dict]:
    """Identify and preserve Q&A pairs in conversation"""
    enhanced = []
    i = 0
    
    while i < len(conversation):
        current = conversation[i]
        content = current.get('content', '').strip()
        
        # Check if this is a question
        if content.endswith('?') and i + 1 < len(conversation):
            next_mem = conversation[i + 1]
            next_content = next_mem.get('content', '').strip()
            
            # If next is likely an answer (longer than question)
            if len(next_content) > len(content) * 1.2:
                # Combine Q&A into single memory
                qa_memory = {
                    'content': f"Q: {content}\nA: {next_content}",
                    'timestamp': current.get('timestamp'),
                    'metadata': {
                        **current.get('metadata', {}),
                        'type': 'qa_pair',
                        'question': content,
                        'answer': next_content
                    }
                }
                enhanced.append(qa_memory)
                i += 2  # Skip both Q and A
                continue
        
        # Not a Q&A pair, add as-is
        enhanced.append(current)
        i += 1
    
    return enhanced

def clean_and_rebuild_memories(input_file: str = "data/chatgpt_memories.json",
                              output_file: str = "data/chatgpt_memories_cleaned.json") -> None:
    """Main function to clean and rebuild memory database"""
    
    print("\n🔧 ChatGPT Memory Rebuilder v1.2.1")
    print("=" * 50)
    
    # Load raw memories
    raw_memories = load_raw_memories(input_file)
    
    # Analyze quality
    print("\n📊 Analyzing memory quality...")
    stats = analyze_memory_quality(raw_memories)
    print(f"  Empty: {stats['empty']:,}")
    print(f"  Tiny (<20): {stats['tiny']:,}")
    print(f"  Small (20-40): {stats['small']:,}")
    print(f"  Medium (40-200): {stats['medium']:,}")
    print(f"  Large (200-1000): {stats['large']:,}")
    print(f"  Huge (>1000): {stats['huge']:,}")
    print(f"  Fragments to merge: {len(stats['fragments_to_merge']):,}")
    
    # Group by conversation
    print("\n🗂️ Grouping memories by conversation (30 min gaps)...")
    conversations = group_memories_by_time(raw_memories, gap_minutes=30)
    print(f"✅ Found {len(conversations):,} conversation groups")
    
    # Process each conversation
    print("\n🔄 Processing conversations...")
    cleaned_memories = []
    
    for i, conv in enumerate(conversations):
        if i % 100 == 0:
            print(f"  Processing conversation {i+1}/{len(conversations)}...")
        
        # Merge fragments
        merged = merge_conversation_fragments(conv, min_length=40)
        
        # Preserve Q&A pairs
        enhanced = preserve_qa_pairs(merged)
        
        # Add to cleaned memories
        cleaned_memories.extend(enhanced)
    
    # Filter out empty memories
    cleaned_memories = [m for m in cleaned_memories if m.get('content', '').strip()]
    
    # Stats on cleaned data
    print("\n📈 Cleaning Results:")
    print(f"  Original memories: {len(raw_memories):,}")
    print(f"  Cleaned memories: {len(cleaned_memories):,}")
    print(f"  Reduction: {len(raw_memories) - len(cleaned_memories):,} ({(1 - len(cleaned_memories)/len(raw_memories))*100:.1f}%)")
    
    # Analyze cleaned quality
    cleaned_stats = analyze_memory_quality(cleaned_memories)
    print(f"\n📊 Cleaned Memory Quality:")
    print(f"  Empty: {cleaned_stats['empty']:,}")
    print(f"  Tiny (<20): {cleaned_stats['tiny']:,}")
    print(f"  Small (20-40): {cleaned_stats['small']:,}")
    print(f"  Medium (40-200): {cleaned_stats['medium']:,}")
    print(f"  Large (200-1000): {cleaned_stats['large']:,}")
    print(f"  Huge (>1000): {cleaned_stats['huge']:,}")
    
    # Save cleaned memories
    print(f"\n💾 Saving cleaned memories to {output_file}...")
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_memories, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(cleaned_memories):,} cleaned memories")
    
    # Now rebuild FAISS index
    print("\n🔨 Rebuilding FAISS index with cleaned memories...")
    rebuild_faiss_index(cleaned_memories)
    
    print("\n✨ Memory rebuild complete!")
    print(f"   Use 'data/chatgpt_memories_cleaned.json' for better search quality")

def rebuild_faiss_index(memories: List[Dict]) -> None:
    """Rebuild FAISS index with cleaned memories"""
    try:
        from storage.faiss_store import FaissVectorStore
        from integrations.embeddings import OpenAIEmbeddings
        import faiss
        import pickle
        
        print("  Initializing FAISS and embeddings...")
        vector_store = FaissVectorStore(dimension=1536)
        embeddings = OpenAIEmbeddings()
        
        print(f"  Generating embeddings for {len(memories):,} memories...")
        batch_size = 100
        
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i+batch_size]
            print(f"    Processing batch {i//batch_size + 1}/{(len(memories) + batch_size - 1)//batch_size}...")
            
            for j, mem in enumerate(batch):
                content = mem.get('content', '')
                if content:
                    try:
                        # Generate embedding
                        embedding = embeddings.embed_text(content)
                        
                        # Add to vector store with embedding
                        vector_store.memories[i+j] = {
                            'content': content,
                            'embedding': embedding,
                            'metadata': mem.get('metadata', {}),
                            'timestamp': mem.get('timestamp', datetime.now().isoformat())
                        }
                        vector_store.index.add(np.array([embedding]))
                    except Exception as e:
                        print(f"      Warning: Failed to process memory {i+j}: {e}")
        
        # Save the new index
        print("\n  Saving FAISS index...")
        faiss.write_index(vector_store.index, "data/faiss_chatgpt_cleaned.index")
        
        # Save memory dictionary
        with open("data/faiss_chatgpt_cleaned.pkl", 'wb') as f:
            pickle.dump(vector_store.memories, f)
        
        print(f"✅ FAISS index rebuilt with {vector_store.index.ntotal} vectors")
        
    except ImportError:
        print("⚠️ FAISS not available, skipping index rebuild")
        print("   Run 'pip install faiss-cpu' to enable FAISS indexing")

if __name__ == "__main__":
    clean_and_rebuild_memories()