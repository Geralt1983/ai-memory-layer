#!/usr/bin/env python3
"""
Test Memory Loading on EC2
Direct test of the memory system without the API
"""

import json
import os
from datetime import datetime
from core.memory_engine import MemoryEngine
from integrations.embeddings import OpenAIEmbeddings
from storage.faiss_store import FaissVectorStore
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_memory_loading():
    """Test loading ChatGPT memories directly"""
    
    print("🧠 Testing ChatGPT Memory Loading")
    print("=" * 40)
    
    # Check files exist
    memory_file = "./data/chatgpt_memories.json"
    faiss_index = "./data/faiss_chatgpt_index.index"
    faiss_pkl = "./data/faiss_chatgpt_index.pkl"
    
    print(f"📁 Checking files:")
    print(f"  Memory JSON: {os.path.exists(memory_file)} ({os.path.getsize(memory_file) if os.path.exists(memory_file) else 0} bytes)")
    print(f"  FAISS index: {os.path.exists(faiss_index)} ({os.path.getsize(faiss_index) if os.path.exists(faiss_index) else 0} bytes)")
    print(f"  FAISS pkl: {os.path.exists(faiss_pkl)} ({os.path.getsize(faiss_pkl) if os.path.exists(faiss_pkl) else 0} bytes)")
    
    # Load JSON directly
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            memories_data = json.load(f)
        print(f"📊 Direct JSON load: {len(memories_data)} memories")
        
        # Show sample
        if memories_data:
            sample = memories_data[0]
            print(f"📝 Sample memory:")
            print(f"  Content: {sample.get('content', '')[:100]}...")
            print(f"  Role: {sample.get('role', 'unknown')}")
            print(f"  Title: {sample.get('title', 'No title')}")
            print(f"  Type: {sample.get('type', 'unknown')}")
            print(f"  Importance: {sample.get('importance', 0)}")
    
    print("\n🔧 Testing Memory Engine...")
    
    # Test API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else 'short'}")
    else:
        print("❌ No API key found")
        return
    
    try:
        # Initialize components
        embeddings = OpenAIEmbeddings(api_key)
        print("✅ OpenAI embeddings initialized")
        
        vector_store = FaissVectorStore(
            dimension=1536,
            index_path="./data/faiss_chatgpt_index"
        )
        print("✅ FAISS vector store initialized")
        
        # Try to initialize memory engine
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embeddings,
            persist_path=memory_file
        )
        print(f"✅ Memory engine initialized")
        print(f"📊 Loaded memories: {len(memory_engine.memories)}")
        
        # Test search if we have memories
        if len(memory_engine.memories) > 0:
            print("\n🔍 Testing search...")
            results = memory_engine.search_memories("python programming", k=3)
            print(f"📝 Search results: {len(results)}")
            
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result.content[:100]}... (score: {result.relevance_score:.3f})")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_loading()