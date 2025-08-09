#!/usr/bin/env python3
"""
Optimized ChatGPT Memory Loader
==============================

This loader efficiently pairs pre-computed FAISS embeddings with memory content
without regenerating embeddings, enabling instant loading of 23,710 ChatGPT memories.

Key optimizations:
1. Loads FAISS index directly (23,710 pre-computed vectors)
2. Pairs vectors with memory content by position/order
3. No re-embedding - uses existing embeddings
4. Memory streaming for large datasets
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
import faiss
import pickle
from datetime import datetime

from core.memory_engine import Memory, MemoryEngine
from storage.faiss_store import FaissVectorStore
from integrations.embeddings import OpenAIEmbeddings


class OptimizedChatGPTLoader:
    """Optimized loader for ChatGPT memories with pre-computed FAISS embeddings"""
    
    def __init__(self, faiss_index_path: str, memory_json_path: str):
        self.faiss_index_path = faiss_index_path
        self.memory_json_path = memory_json_path
        
    def verify_data_integrity(self) -> bool:
        """Verify that FAISS index and memory JSON have matching counts"""
        print("🔍 Verifying data integrity...")
        
        # Check FAISS index
        try:
            index = faiss.read_index(f"{self.faiss_index_path}.index")
            faiss_count = index.ntotal
            print(f"✅ FAISS index: {faiss_count} vectors")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            return False
            
        # Check memory JSON
        try:
            with open(self.memory_json_path, 'r') as f:
                memories = json.load(f)
                memory_count = len(memories)
                print(f"✅ Memory JSON: {memory_count} memories")
        except Exception as e:
            print(f"❌ Error loading memory JSON: {e}")
            return False
            
        # Verify counts match
        if faiss_count == memory_count:
            print(f"✅ Data integrity verified: {faiss_count} records")
            return True
        else:
            print(f"❌ Count mismatch: FAISS={faiss_count}, JSON={memory_count}")
            return False
    
    def create_optimized_memory_engine(self, embedding_provider: Optional[Any] = None) -> MemoryEngine:
        """Create MemoryEngine with pre-loaded FAISS vectors and paired content"""
        
        print("🚀 Creating optimized memory engine...")
        start_time = time.time()
        
        if not self.verify_data_integrity():
            raise ValueError("Data integrity check failed")
        
        # Create custom FAISS vector store that loads existing data
        vector_store = OptimizedFaissVectorStore(
            dimension=1536,
            index_path=self.faiss_index_path
        )
        
        print(f"📦 Loaded FAISS index: {vector_store.index.ntotal} vectors")
        
        # Load and pair memory content
        print("📚 Loading memory content...")
        with open(self.memory_json_path, 'r') as f:
            memory_data = json.load(f)
        
        # Create Memory objects without embeddings (they're in FAISS)
        memories = []
        for i, mem_data in enumerate(memory_data):
            memory = Memory(
                content=mem_data.get('content', ''),
                metadata=mem_data.get('metadata', {}),
                timestamp=datetime.fromisoformat(mem_data['timestamp'].replace('Z', '+00:00')) if 'timestamp' in mem_data else datetime.now(),
                embedding=None  # Embeddings are in FAISS, indexed by position
            )
            
            # Add additional fields from ChatGPT import
            memory.role = mem_data.get('role', 'user')
            memory.thread_id = mem_data.get('thread_id', '')
            memory.title = mem_data.get('title', '')
            memory.type = mem_data.get('type', 'history')
            memory.importance = mem_data.get('importance', 0.0)
            memory.relevance_score = mem_data.get('relevance_score', 0.0)
            
            memories.append(memory)
            
            # Progress indicator
            if (i + 1) % 5000 == 0:
                print(f"  📝 Processed {i + 1}/{len(memory_data)} memories")
        
        print(f"✅ Created {len(memories)} Memory objects")
        
        # Update vector store with paired memories
        vector_store.memories = {i: memory for i, memory in enumerate(memories)}
        vector_store.current_id = len(memories)
        
        # Create memory engine
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            persist_path=None  # Don't auto-save, we're loading existing data
        )
        
        # Directly set the memories to skip loading from persistence
        memory_engine.memories = memories
        
        load_time = time.time() - start_time
        print(f"🎉 Optimized memory engine ready in {load_time:.2f}s")
        print(f"📊 Stats: {len(memories)} memories, {vector_store.index.ntotal} vectors")
        
        return memory_engine


class OptimizedFaissVectorStore(FaissVectorStore):
    """Optimized FAISS store that loads existing index without regenerating"""
    
    def __init__(self, dimension: int = 1536, index_path: Optional[str] = None):
        self.dimension = dimension
        self.memories: Dict[int, Memory] = {}
        self.current_id = 0
        self.index_path = index_path
        
        # Load existing FAISS index directly
        if index_path and os.path.exists(f"{index_path}.index"):
            print(f"📥 Loading existing FAISS index from {index_path}")
            self.index = faiss.read_index(f"{index_path}.index")
            print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load pickle metadata if available
            if os.path.exists(f"{index_path}.pkl"):
                try:
                    with open(f"{index_path}.pkl", "rb") as f:
                        data = pickle.load(f)
                        self.current_id = data.get("current_id", 0)
                        print(f"📋 Loaded metadata: current_id={self.current_id}")
                except Exception as e:
                    print(f"⚠️ Could not load pickle metadata: {e}")
        else:
            # Fallback to empty index
            print("⚠️ No existing index found, creating empty index")
            self.index = faiss.IndexFlatL2(dimension)


def create_optimized_api(
    faiss_index_path: str = "./data/faiss_chatgpt_index",
    memory_json_path: str = "./data/chatgpt_memories.json",
    openai_api_key: Optional[str] = None
) -> MemoryEngine:
    """
    Create optimized memory engine for API use
    
    Args:
        faiss_index_path: Path to FAISS index files (without extension)
        memory_json_path: Path to memory JSON file
        openai_api_key: OpenAI API key for new embeddings (optional)
    
    Returns:
        Optimized MemoryEngine ready for production use
    """
    
    # Initialize embedding provider (for new memories only)
    embedding_provider = None
    if openai_api_key:
        embedding_provider = OpenAIEmbeddings(api_key=openai_api_key)
    
    # Create optimized loader
    loader = OptimizedChatGPTLoader(faiss_index_path, memory_json_path)
    
    # Create and return optimized memory engine
    return loader.create_optimized_memory_engine(embedding_provider)


if __name__ == "__main__":
    print("🔧 Testing Optimized ChatGPT Loader")
    print("=" * 50)
    
    # Test the loader
    try:
        faiss_path = "./data/faiss_chatgpt_index"
        json_path = "./data/chatgpt_memories.json"
        
        loader = OptimizedChatGPTLoader(faiss_path, json_path)
        
        # Verify integrity
        if loader.verify_data_integrity():
            print("\n🚀 Creating optimized memory engine...")
            memory_engine = loader.create_optimized_memory_engine()
            
            print("\n📊 Testing search functionality...")
            results = memory_engine.search_memories("python langchain", top_k=3)
            print(f"Found {len(results)} results")
            
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.content[:100]}...")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()