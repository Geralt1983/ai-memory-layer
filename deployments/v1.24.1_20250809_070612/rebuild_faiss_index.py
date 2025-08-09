#!/usr/bin/env python3
"""
FAISS Index Rebuilder for ChatGPT Memories
==========================================

One-time script to rebuild the FAISS index from the full chatgpt_memories.json
ensuring perfect alignment between memory content and vector embeddings.

This solves the 30/32 memory limit issue by rebuilding the complete index
with all 23,710 ChatGPT conversation memories.
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def rebuild_faiss_index():
    """Rebuild FAISS index from complete ChatGPT memory data"""
    
    print("🔧 FAISS Index Rebuilder for ChatGPT Memories")
    print("=" * 50)
    
    # Configuration
    MEMORY_JSON = "./data/chatgpt_memories.json"
    FAISS_INDEX_PATH = "./data/faiss_chatgpt_index"
    
    # Verify environment
    if not os.path.exists(".env"):
        print("⚠️  No .env file found - make sure OPENAI_API_KEY is set")
    else:
        from dotenv import load_dotenv
        load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required for embedding generation")
        sys.exit(1)
    
    # Check input file
    if not os.path.exists(MEMORY_JSON):
        print(f"❌ Memory JSON not found: {MEMORY_JSON}")
        sys.exit(1)
    
    file_size = os.path.getsize(MEMORY_JSON) / 1024 / 1024
    print(f"📊 Memory file: {MEMORY_JSON} ({file_size:.1f}MB)")
    
    # Load memory data
    print("📚 Loading ChatGPT memory data...")
    start_time = time.time()
    
    try:
        with open(MEMORY_JSON, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        load_time = time.time() - start_time
        print(f"✅ Loaded {len(memory_data)} memories in {load_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error loading memory JSON: {e}")
        sys.exit(1)
    
    # Initialize components
    print("🔧 Initializing FAISS vector store...")
    
    try:
        from storage.faiss_store import FaissVectorStore
        from integrations.embeddings import OpenAIEmbeddings
        from core.memory_engine import Memory
        
        # Create fresh vector store
        vector_store = FaissVectorStore(dimension=1536, index_path=None)  # Fresh instance
        embedding_provider = OpenAIEmbeddings()
        
        print(f"✅ Vector store initialized (dimension: {vector_store.dimension})")
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        sys.exit(1)
    
    # Process memories in batches
    print("🚀 Building FAISS index with embeddings...")
    rebuild_start = time.time()
    
    batch_size = 100
    total_processed = 0
    errors = 0
    
    for i in range(0, len(memory_data), batch_size):
        batch = memory_data[i:i + batch_size]
        batch_start = time.time()
        
        for j, mem_data in enumerate(batch):
            try:
                # Parse timestamp
                timestamp_str = mem_data.get('timestamp')
                if timestamp_str:
                    try:
                        if 'T' in timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.fromtimestamp(float(timestamp_str))
                        timestamp = timestamp.replace(tzinfo=None)
                    except:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                
                # Create Memory object
                memory = Memory(
                    content=mem_data.get('content', ''),
                    embedding=None,  # Will be generated
                    metadata=mem_data.get('metadata', {}),
                    timestamp=timestamp,
                    relevance_score=mem_data.get('relevance_score', 0.0)
                )
                
                # Add ChatGPT-specific fields
                memory.role = mem_data.get('role', 'user')
                memory.thread_id = mem_data.get('thread_id', '')
                memory.title = mem_data.get('title', '')
                memory.type = mem_data.get('type', 'chatgpt_history')
                memory.importance = mem_data.get('importance', 1.0)
                
                # Generate embedding and add to vector store
                if memory.content.strip():  # Only process non-empty content
                    embedding = embedding_provider.embed_text(memory.content)
                    memory.embedding = embedding
                    
                    # Add to vector store
                    memory_id = vector_store.add_memory(memory)
                    total_processed += 1
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                if errors <= 5:  # Log first 5 errors
                    print(f"⚠️  Error processing memory {i+j}: {e}")
                continue
        
        batch_time = time.time() - batch_start
        progress = (i + len(batch)) / len(memory_data) * 100
        
        print(f"📝 Processed {i + len(batch):,}/{len(memory_data):,} memories "
              f"({progress:.1f}%) - Batch time: {batch_time:.2f}s")
    
    rebuild_time = time.time() - rebuild_start
    
    # Save the index
    print("💾 Saving FAISS index and metadata...")
    save_start = time.time()
    
    try:
        vector_store.save_index(FAISS_INDEX_PATH)
        save_time = time.time() - save_start
        
        print(f"✅ FAISS index saved to: {FAISS_INDEX_PATH}")
        print(f"💾 Save time: {save_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error saving index: {e}")
        sys.exit(1)
    
    # Final verification
    print("🔍 Verifying rebuilt index...")
    try:
        import faiss
        index = faiss.read_index(f"{FAISS_INDEX_PATH}.index")
        
        print(f"✅ Verification complete!")
        print(f"📊 Final Statistics:")
        print(f"   • Total memories processed: {total_processed:,}")
        print(f"   • Errors encountered: {errors:,}")
        print(f"   • FAISS index vectors: {index.ntotal:,}")
        print(f"   • Total rebuild time: {rebuild_time:.2f}s")
        print(f"   • Processing rate: {total_processed/rebuild_time:.0f} memories/sec")
        
        if index.ntotal != total_processed:
            print(f"⚠️  Warning: Vector count ({index.ntotal}) != processed count ({total_processed})")
        else:
            print("✅ Perfect alignment: vectors match processed memories")
            
    except Exception as e:
        print(f"⚠️  Could not verify index: {e}")
    
    print(f"\n🎉 FAISS index rebuild complete!")
    print(f"📁 Files created:")
    print(f"   • {FAISS_INDEX_PATH}.index")
    print(f"   • {FAISS_INDEX_PATH}.pkl")
    print(f"\n🚀 Ready to launch API with full {total_processed:,} ChatGPT memories!")


if __name__ == "__main__":
    rebuild_faiss_index()