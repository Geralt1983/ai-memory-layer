#!/usr/bin/env python3
"""
Build FAISS index for cleaned memories - optimized version
"""
import json
import sys
import os
import time
import pickle
from pathlib import Path
sys.path.append('.')

# Load environment
from dotenv import load_dotenv
load_dotenv('.env.local')

def build_cleaned_faiss_index():
    """Build FAISS index specifically for cleaned memories"""
    
    print("🧹 Building FAISS Index for 21,338 Cleaned Memories")
    print("=" * 60)
    
    # Check files
    cleaned_file = "data/chatgpt_memories_cleaned.json"
    if not Path(cleaned_file).exists():
        print(f"❌ Cleaned memory file not found: {cleaned_file}")
        return False
    
    # Load cleaned memories
    print("📄 Loading cleaned memories...")
    start_time = time.time()
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        memories = json.load(f)
    
    load_time = time.time() - start_time
    print(f"✅ Loaded {len(memories):,} cleaned memories in {load_time:.2f}s")
    
    # Initialize OpenAI client
    print("🔧 Initializing OpenAI embeddings...")
    try:
        import openai
        client = openai.OpenAI()
        
        # Test connection
        print("🧪 Testing OpenAI connection...")
        test_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="test"
        )
        print("✅ OpenAI connection successful")
    except Exception as e:
        print(f"❌ OpenAI setup failed: {e}")
        return False
    
    # Initialize FAISS
    print("🔧 Initializing FAISS...")
    try:
        import faiss
        import numpy as np
        
        # Create index
        dimension = 1536  # OpenAI ada-002 dimension
        index = faiss.IndexFlatL2(dimension)
        print(f"✅ FAISS index created (dimension: {dimension})")
    except Exception as e:
        print(f"❌ FAISS setup failed: {e}")
        return False
    
    # Process memories in batches
    batch_size = 20  # Smaller batches for rate limiting
    total = len(memories)
    all_embeddings = []
    memory_map = {}
    
    print(f"🚀 Processing {total:,} memories in batches of {batch_size}...")
    print("⏱️  This will take ~10-15 minutes due to OpenAI rate limits...")
    
    for i in range(0, total, batch_size):
        batch = memories[i:i+batch_size]
        batch_start = time.time()
        
        try:
            # Prepare batch texts
            batch_texts = []
            batch_indices = []
            
            for j, mem in enumerate(batch):
                content = mem['content'].strip()
                if len(content) > 0:  # Skip empty
                    batch_texts.append(content)
                    batch_indices.append(i + j)
            
            if not batch_texts:
                continue
            
            # Get embeddings for batch
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch_texts
            )
            
            # Process embeddings
            for idx, (text, mem_idx) in enumerate(zip(batch_texts, batch_indices)):
                embedding = response.data[idx].embedding
                all_embeddings.append(embedding)
                memory_map[len(all_embeddings) - 1] = {
                    'content': text,
                    'original_index': mem_idx,
                    'metadata': memories[mem_idx].get('metadata', {})
                }
            
            batch_time = time.time() - batch_start
            progress = min((i + len(batch)) / total * 100, 100)
            processed = len(all_embeddings)
            
            if batch_time > 0:
                rate = len(batch_texts) / batch_time
                eta_seconds = (total - processed) / rate if rate > 0 else 0
                eta_minutes = int(eta_seconds / 60)
                
                print(f"📝 Batch {i//batch_size + 1}: {processed:,}/{total:,} ({progress:.1f}%) - "
                      f"Rate: {rate:.1f}/sec - ETA: {eta_minutes}min")
            
            # Rate limiting delay
            if i + batch_size < total:
                time.sleep(0.5)  # Small delay between batches
                
        except Exception as e:
            print(f"⚠️ Error in batch {i}-{i+len(batch)}: {e}")
            time.sleep(2)  # Longer delay on error
            continue
    
    if not all_embeddings:
        print("❌ No embeddings generated")
        return False
    
    processing_time = time.time() - start_time
    print(f"\n✅ Generated {len(all_embeddings):,} embeddings in {processing_time/60:.1f} minutes")
    
    # Add embeddings to FAISS index
    print("📊 Building FAISS index...")
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    index.add(embeddings_array)
    
    print(f"✅ FAISS index built with {index.ntotal:,} vectors")
    
    # Save files
    print("💾 Saving cleaned FAISS index files...")
    try:
        # Save FAISS index
        faiss.write_index(index, "data/faiss_chatgpt_cleaned.index")
        
        # Save memory mapping
        with open("data/faiss_chatgpt_cleaned.pkl", 'wb') as f:
            pickle.dump(memory_map, f)
        
        print("✅ Cleaned FAISS index saved successfully!")
        
        # Verification
        print(f"📊 Final Statistics:")
        print(f"   • Original memories: {total:,}")
        print(f"   • Processed memories: {len(all_embeddings):,}")
        print(f"   • FAISS vectors: {index.ntotal:,}")
        print(f"   • Processing time: {processing_time/60:.1f} minutes")
        print(f"   • Average rate: {len(all_embeddings)/(processing_time/60):.0f} memories/min")
        
        print(f"\n📁 Files created:")
        print(f"   • data/faiss_chatgpt_cleaned.index ({os.path.getsize('data/faiss_chatgpt_cleaned.index')/1024/1024:.1f}MB)")
        print(f"   • data/faiss_chatgpt_cleaned.pkl ({os.path.getsize('data/faiss_chatgpt_cleaned.pkl')/1024/1024:.1f}MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Building proper FAISS index for cleaned memories...")
    success = build_cleaned_faiss_index()
    
    if success:
        print("\n🎉 SUCCESS: Cleaned FAISS index is ready!")
        print("🔄 Next step: Restart the API to use the new index")
        print("📊 Expected improvement: Better search relevance, no more fragments")
    else:
        print("\n❌ FAILED: Could not build cleaned FAISS index")