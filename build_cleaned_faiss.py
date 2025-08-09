#!/usr/bin/env python3
"""
Build FAISS index specifically for cleaned memories
"""
import json
import sys
import time
from pathlib import Path
sys.path.append('.')

from integrations.embeddings_factory import get_embedder
from storage.faiss_store import FaissVectorStore

def build_cleaned_index():
    """Build FAISS index from cleaned memories"""
    
    print("🧹 Building FAISS Index for Cleaned Memories")
    print("=" * 50)
    
    # Load cleaned memories
    cleaned_file = "data/chatgpt_memories_cleaned.json"
    if not Path(cleaned_file).exists():
        print(f"❌ Cleaned memory file not found: {cleaned_file}")
        return False
    
    print(f"📄 Loading cleaned memories...")
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        memories = json.load(f)
    
    print(f"✅ Loaded {len(memories):,} cleaned memories")
    
    # Initialize components
    print("🔧 Initializing embeddings and vector store...")
    # Use factory pattern for embedding provider
    # Use source-specific routing for ChatGPT memories (obsidian tag)
    from integrations.embeddings_factory import get_embedder_for
    embeddings = get_embedder_for("obsidian")
    vector_store = FaissVectorStore(embeddings, dimension=1536)
    
    # Process memories in batches
    batch_size = 50
    total = len(memories)
    start_time = time.time()
    
    print(f"🚀 Processing {total:,} memories in batches of {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch = memories[i:i+batch_size]
        batch_start = time.time()
        
        try:
            # Add batch to vector store
            for j, mem in enumerate(batch):
                content = mem['content']
                if len(content.strip()) > 0:  # Skip empty
                    vector_store.add_memory_content(content, f"mem_{i+j}")
            
            batch_time = time.time() - batch_start
            progress = (i + len(batch)) / total * 100
            rate = len(batch) / batch_time if batch_time > 0 else 0
            
            print(f"📝 Processed {i + len(batch):,}/{total:,} ({progress:.1f}%) - "
                  f"Rate: {rate:.1f} memories/sec")
                  
        except Exception as e:
            print(f"⚠️ Error in batch {i}-{i+len(batch)}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Save index
    print("💾 Saving cleaned FAISS index...")
    try:
        vector_store.save_index("data/faiss_chatgpt_cleaned")
        print("✅ Cleaned FAISS index saved successfully!")
        
        # Verify
        if hasattr(vector_store, 'index'):
            print(f"📊 Final verification:")
            print(f"   • Memories processed: {total:,}")
            print(f"   • FAISS vectors: {vector_store.index.ntotal:,}")
            print(f"   • Total time: {total_time:.1f}s")
            print(f"   • Processing rate: {total/total_time:.0f} memories/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving index: {e}")
        return False

if __name__ == "__main__":
    success = build_cleaned_index()
    if success:
        print("\n🎉 Cleaned FAISS index ready!")
        print("🔄 Restart the API to use the cleaned index")
    else:
        print("\n❌ Failed to build cleaned FAISS index")