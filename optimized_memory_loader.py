#!/usr/bin/env python3
"""
Optimized Memory Loader for ChatGPT Data
========================================

This module provides efficient loading of ChatGPT memories with proper
synchronization between FAISS index and memory content. Designed to work
with the rebuilt index from rebuild_faiss_index.py.
"""

import json
import os
import sys
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class OptimizedChatGPTLoader:
    """Efficiently loads ChatGPT memories with synchronized FAISS index"""
    
    def __init__(self, 
                 faiss_index_path: str = "./data/faiss_chatgpt_index",
                 memory_json_path: str = "./data/chatgpt_memories.json",
                 openai_api_key: Optional[str] = None):
        
        self.faiss_index_path = faiss_index_path
        self.memory_json_path = memory_json_path
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize components
        self.faiss_index = None
        self.memory_data = None
        self.memory_engine = None
        
        # Performance tracking
        self.load_start_time = None
        self.stats = {}
        
    def verify_data_integrity(self) -> bool:
        """Verify that FAISS index and memory data are properly aligned"""
        
        print("🔍 Verifying data integrity...")
        
        # Check file existence
        index_file = f"{self.faiss_index_path}.index"
        pkl_file = f"{self.faiss_index_path}.pkl"
        
        if not os.path.exists(index_file):
            print(f"❌ FAISS index not found: {index_file}")
            return False
            
        if not os.path.exists(pkl_file):
            print(f"❌ FAISS metadata not found: {pkl_file}")
            return False
            
        if not os.path.exists(self.memory_json_path):
            print(f"❌ Memory JSON not found: {self.memory_json_path}")
            return False
        
        # Load and compare counts
        try:
            import faiss
            index = faiss.read_index(index_file)
            
            with open(self.memory_json_path, 'r') as f:
                memory_data = json.load(f)
            
            index_count = index.ntotal
            memory_count = len(memory_data)
            
            print(f"✅ FAISS index vectors: {index_count:,}")
            print(f"✅ Memory JSON entries: {memory_count:,}")
            
            if abs(index_count - memory_count) > 100:  # Allow small discrepancy
                print(f"⚠️  Warning: Count mismatch > 100 ({abs(index_count - memory_count)})")
                return False
            
            if index_count == 0:
                print("❌ FAISS index is empty")
                return False
                
            print("✅ Data integrity verification passed")
            return True
            
        except Exception as e:
            print(f"❌ Integrity check failed: {e}")
            return False
        
    def load_faiss_components(self):
        """Load FAISS index and metadata efficiently"""
        
        print("📚 Loading FAISS components...")
        start_time = time.time()
        
        try:
            import faiss
            
            # Load FAISS index
            index_file = f"{self.faiss_index_path}.index"
            self.faiss_index = faiss.read_index(index_file)
            
            # Load metadata (if exists)
            pkl_file = f"{self.faiss_index_path}.pkl"
            metadata = None
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    metadata = pickle.load(f)
            
            load_time = time.time() - start_time
            
            print(f"✅ FAISS loaded: {self.faiss_index.ntotal:,} vectors in {load_time:.2f}s")
            self.stats['faiss_load_time'] = load_time
            self.stats['faiss_vectors'] = self.faiss_index.ntotal
            
            return metadata
            
        except Exception as e:
            print(f"❌ FAISS loading failed: {e}")
            raise
        
    def load_memory_data(self):
        """Load ChatGPT memory JSON efficiently"""
        
        print("📄 Loading ChatGPT memory data...")
        start_time = time.time()
        
        try:
            file_size = os.path.getsize(self.memory_json_path) / 1024 / 1024
            print(f"📊 File size: {file_size:.1f}MB")
            
            with open(self.memory_json_path, 'r', encoding='utf-8') as f:
                self.memory_data = json.load(f)
            
            load_time = time.time() - start_time
            
            print(f"✅ Memory data loaded: {len(self.memory_data):,} entries in {load_time:.2f}s")
            self.stats['memory_load_time'] = load_time
            self.stats['memory_count'] = len(self.memory_data)
            
        except Exception as e:
            print(f"❌ Memory data loading failed: {e}")
            raise
    
    def create_memory_engine(self):
        """Create synchronized memory engine with ChatGPT data"""
        
        print("🔧 Creating synchronized memory engine...")
        start_time = time.time()
        
        try:
            from core.memory_engine import MemoryEngine, Memory
            from storage.faiss_store import FaissVectorStore
            from integrations.embeddings import OpenAIEmbeddings
            
            # Create vector store from pre-loaded FAISS index
            vector_store = FaissVectorStore(
                dimension=1536,
                index_path=None  # Don't auto-load, we'll set manually
            )
            
            # Set the pre-loaded index
            vector_store.index = self.faiss_index
            vector_store.memories = []  # Initialize empty list for consistency
            
            # Create embedding provider
            embedding_provider = OpenAIEmbeddings(api_key=self.openai_api_key)
            
            # Create memory engine
            self.memory_engine = MemoryEngine(
                vector_store=vector_store,
                embedding_provider=embedding_provider
            )
            
            # Convert JSON data to Memory objects and sync with engine
            print("🔄 Converting and syncing memory objects...")
            conversion_start = time.time()
            
            memories_converted = 0
            for i, mem_data in enumerate(self.memory_data):
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
                        embedding=None,  # Embedding is in FAISS
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
                    
                    # Add to memory engine's memory list (for get_stats, etc.)
                    self.memory_engine.memories.append(memory)
                    memories_converted += 1
                    
                except Exception as e:
                    if memories_converted < 5:  # Log first few errors
                        print(f"⚠️  Error converting memory {i}: {e}")
                    continue
            
            conversion_time = time.time() - conversion_start
            create_time = time.time() - start_time
            
            print(f"✅ Memory engine created with {memories_converted:,} memories")
            print(f"🔄 Conversion time: {conversion_time:.2f}s")
            print(f"⚡ Total engine creation time: {create_time:.2f}s")
            
            self.stats['conversion_time'] = conversion_time
            self.stats['create_time'] = create_time
            self.stats['memories_converted'] = memories_converted
            
        except Exception as e:
            print(f"❌ Memory engine creation failed: {e}")
            raise
    
    def load_complete_system(self):
        """Load complete ChatGPT memory system with all components"""
        
        print("🚀 Loading Complete ChatGPT Memory System")
        print("=" * 50)
        self.load_start_time = time.time()
        
        # Step 1: Verify data integrity
        if not self.verify_data_integrity():
            raise RuntimeError("Data integrity check failed")
        
        # Step 2: Load FAISS components
        self.load_faiss_components()
        
        # Step 3: Load memory data
        self.load_memory_data()
        
        # Step 4: Create synchronized memory engine
        self.create_memory_engine()
        
        total_time = time.time() - self.load_start_time
        
        print("🎉 System Loading Complete!")
        print("=" * 50)
        print(f"📊 Final Statistics:")
        print(f"   • FAISS vectors: {self.stats['faiss_vectors']:,}")
        print(f"   • Memory entries: {self.stats['memory_count']:,}")
        print(f"   • Memories converted: {self.stats['memories_converted']:,}")
        print(f"   • Total load time: {total_time:.2f}s")
        print(f"   • Average rate: {self.stats['memories_converted']/total_time:.0f} memories/sec")
        
        # Verify final state
        engine_memory_count = len(self.memory_engine.memories) if self.memory_engine else 0
        print(f"   • Engine memory count: {engine_memory_count:,}")
        
        if engine_memory_count > 20000:  # Expect around 23,710
            print("✅ SUCCESS: Full ChatGPT dataset loaded")
        else:
            print(f"⚠️  Warning: Lower memory count than expected")
        
        return self.memory_engine
    
    def get_loading_stats(self):
        """Return detailed loading statistics"""
        return {
            **self.stats,
            'total_load_time': time.time() - self.load_start_time if self.load_start_time else 0,
            'engine_ready': self.memory_engine is not None,
            'timestamp': datetime.now().isoformat()
        }


def create_optimized_chatgpt_engine(
    faiss_index_path: str = "./data/faiss_chatgpt_index",
    memory_json_path: str = "./data/chatgpt_memories.json",
    openai_api_key: Optional[str] = None
):
    """
    Main entry point for loading optimized ChatGPT memory engine
    
    This function handles the complete loading process and returns a ready-to-use
    memory engine with all 23,710+ ChatGPT conversation memories.
    """
    
    loader = OptimizedChatGPTLoader(
        faiss_index_path=faiss_index_path,
        memory_json_path=memory_json_path,
        openai_api_key=openai_api_key
    )
    
    try:
        memory_engine = loader.load_complete_system()
        
        print(f"\n🎯 Ready to serve {len(memory_engine.memories):,} ChatGPT memories!")
        return memory_engine
        
    except Exception as e:
        print(f"❌ Failed to create optimized ChatGPT engine: {e}")
        print("💡 Try running rebuild_faiss_index.py first to ensure proper data alignment")
        raise


if __name__ == "__main__":
    """Direct testing of the optimized loader"""
    
    # Load environment
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv()
    except ImportError:
        pass
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required for testing")
        sys.exit(1)
    
    print("🧪 Testing Optimized ChatGPT Memory Loader")
    print("=" * 50)
    
    try:
        engine = create_optimized_chatgpt_engine()
        
        # Quick verification
        print("\n🧪 Quick Functionality Test:")
        stats = engine.get_stats()
        print(f"📊 Engine stats: {stats}")
        
        # Test search
        if len(engine.memories) > 0:
            results = engine.search_memories("python programming", top_k=3)
            print(f"🔍 Search test returned {len(results)} results")
            
            if results:
                print("📝 Sample result:")
                print(f"   Content: {results[0].content[:100]}...")
                print(f"   Relevance: {results[0].relevance_score:.3f}")
        
        print("\n✅ Optimized loader test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)