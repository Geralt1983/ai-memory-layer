#!/usr/bin/env python3
"""
Fixed Compatible Memory Loader
Works with the existing FaissVectorStore interface that returns List[Memory]
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from core.memory_engine import Memory, MemoryEngine
from integrations.embeddings import OpenAIEmbeddings
from storage.faiss_store import FaissVectorStore
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedCompatibleMemoryLoader:
    """
    Fixed memory loader that works with existing FaissVectorStore interface
    Uses the correct search method that returns List[Memory] directly
    """
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        
    def load_chatgpt_memories_compatible(
        self, 
        memory_json_path: str,
        faiss_index_path: str
    ) -> MemoryEngine:
        """
        Load ChatGPT memories compatible with deployed Memory class
        
        Args:
            memory_json_path: Path to ChatGPT memories JSON
            faiss_index_path: Path to FAISS index (without .index extension)
            
        Returns:
            MemoryEngine with loaded memories
        """
        logger.info("🚀 Starting fixed compatible ChatGPT memory loading...")
        
        # 1. Load memory JSON data
        logger.info(f"📄 Loading memories from {memory_json_path}")
        with open(memory_json_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        logger.info(f"✅ Loaded {len(memory_data)} memories from JSON")
        
        # 2. Initialize components
        embeddings_provider = None
        if self.api_key:
            embeddings_provider = OpenAIEmbeddings(self.api_key)
            logger.info("✅ OpenAI embeddings provider ready")
        
        # Create FAISS vector store
        vector_store = FaissVectorStore(
            dimension=1536,
            index_path=faiss_index_path
        )
        logger.info("✅ FAISS vector store initialized")
        
        # Check FAISS index status
        if hasattr(vector_store.index, 'ntotal'):
            logger.info(f"✅ FAISS index loaded with {vector_store.index.ntotal} vectors")
        
        # Create memory engine (don't load from persist_path to avoid overwrite)
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embeddings_provider,
            persist_path=None  # Don't auto-load/save
        )
        
        # 3. Convert JSON to Memory objects (compatible version)
        logger.info("🔄 Converting to compatible Memory objects...")
        memories = []
        
        batch_size = 1000
        for i in range(0, len(memory_data), batch_size):
            batch = memory_data[i:i + batch_size]
            
            for j, mem_data in enumerate(batch):
                try:
                    # Parse timestamp
                    timestamp = self._parse_timestamp(mem_data.get('timestamp'))
                    
                    # Store enhanced fields in metadata (compatible approach)
                    enhanced_metadata = {
                        **mem_data.get('metadata', {}),
                        # ChatGPT specific fields
                        'role': mem_data.get('role', 'user'),
                        'thread_id': mem_data.get('thread_id'),
                        'title': mem_data.get('title'),
                        'type': mem_data.get('type', 'history'),
                        'importance': mem_data.get('importance', 1.0),
                        'source': 'chatgpt_import'
                    }
                    
                    # Create compatible Memory object
                    memory = Memory(
                        content=mem_data.get('content', ''),
                        embedding=None,  # Will use FAISS index
                        metadata=enhanced_metadata,
                        timestamp=timestamp,
                        relevance_score=0.0
                    )
                    
                    memories.append(memory)
                    
                except Exception as e:
                    logger.error(f"Error processing memory {i + j}: {e}")
                    continue
            
            logger.info(f"Processed batch {i // batch_size + 1}/{(len(memory_data) + batch_size - 1) // batch_size}")
        
        # 4. Set memories directly in engine
        memory_engine.memories = memories
        logger.info(f"✅ Successfully loaded {len(memories)} compatible memories")
        
        return memory_engine
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp string efficiently"""
        if not timestamp_str:
            return datetime.now()
        
        try:
            if 'T' in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).replace(tzinfo=None)
            else:
                return datetime.fromtimestamp(float(timestamp_str))
        except:
            return datetime.now()
    
    def enhanced_search_memories(
        self, 
        memory_engine: MemoryEngine, 
        query: str, 
        k: int = 5
    ) -> List[Memory]:
        """
        Enhanced search using the correct FaissVectorStore interface
        Works with the existing search method that returns List[Memory]
        """
        if not memory_engine.memories:
            logger.warning("No memories available for search")
            return []
        
        if not memory_engine.embedding_provider:
            logger.error("No embedding provider available")
            return []
        
        try:
            # Use the standard MemoryEngine search_memories method
            # This will work with the existing FaissVectorStore interface
            results = memory_engine.search_memories(query, k)
            
            # Apply additional importance weighting from metadata
            enhanced_results = []
            for memory in results:
                try:
                    # Get importance from metadata
                    importance = memory.metadata.get('importance', 1.0)
                    memory_type = memory.metadata.get('type', 'history')
                    
                    # Apply additional weighting to the existing relevance_score
                    base_score = memory.relevance_score
                    
                    # Type-based boosting
                    type_multiplier = {
                        'correction': 1.5,
                        'summary': 1.2,
                        'identity': 1.1,
                        'history': 1.0
                    }.get(memory_type, 1.0)
                    
                    # Age decay
                    age_days = (datetime.now() - memory.timestamp).days
                    age_factor = 0.5 ** (age_days / 30.0) if age_days > 0 else 1.0
                    
                    # Enhanced score
                    enhanced_score = base_score * importance * type_multiplier * age_factor
                    memory.relevance_score = enhanced_score
                    
                    enhanced_results.append(memory)
                    
                except Exception as e:
                    logger.error(f"Error enhancing memory score: {e}")
                    enhanced_results.append(memory)  # Add with original score
            
            # Re-sort by enhanced scores
            enhanced_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return enhanced_results[:k]
            
        except Exception as e:
            logger.error(f"Enhanced search error: {e}")
            return []
    
    def test_system(self, memory_engine: MemoryEngine):
        """Test the fixed compatible memory system"""
        logger.info("🔍 Testing fixed compatible memory system...")
        
        test_queries = [
            "python programming",
            "LangChain OpenAI", 
            "vector database",
            "ChatGPT conversation",
            "memory search"
        ]
        
        for query in test_queries:
            try:
                start_time = datetime.now()
                results = self.enhanced_search_memories(memory_engine, query, k=3)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                logger.info(f"✅ '{query}': {len(results)} results in {duration:.3f}s")
                
                if results:
                    top = results[0]
                    role = top.metadata.get('role', 'unknown')
                    title = top.metadata.get('title', 'No title')
                    logger.info(f"   📝 Top [{role}]: {title} (score: {top.relevance_score:.3f})")
                    logger.info(f"       Content: {top.content[:100]}...")
                
            except Exception as e:
                logger.error(f"Test search failed for '{query}': {e}")

def main():
    """Test the fixed compatible loader"""
    if len(sys.argv) < 3:
        print("Usage: python fixed_compatible_loader.py <memory.json> <faiss_index_path>")
        print("Example: python fixed_compatible_loader.py data/chatgpt_memories.json data/faiss_chatgpt_index")
        sys.exit(1)
    
    memory_json = sys.argv[1]
    faiss_index = sys.argv[2]
    
    # Verify files exist
    if not os.path.exists(memory_json):
        logger.error(f"Memory file not found: {memory_json}")
        sys.exit(1)
    
    if not os.path.exists(f"{faiss_index}.index"):
        logger.error(f"FAISS index not found: {faiss_index}.index")
        sys.exit(1)
    
    # Create loader and test
    loader = FixedCompatibleMemoryLoader()
    
    try:
        memory_engine = loader.load_chatgpt_memories_compatible(memory_json, faiss_index)
        
        if memory_engine and len(memory_engine.memories) > 0:
            logger.info(f"🎉 Fixed compatible system ready with {len(memory_engine.memories)} memories!")
            
            # Test the system
            loader.test_system(memory_engine)
            
            return memory_engine
        else:
            logger.error("❌ Failed to create fixed compatible memory system")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()