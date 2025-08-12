#!/usr/bin/env python3
"""
Optimized Memory Loader - Uses Pre-computed FAISS Embeddings
Based on 2025 best practices for FAISS vector database optimization
"""

import json
import os
import sys
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from core.memory_engine import Memory
from integrations.embeddings import OpenAIEmbeddings
from storage.faiss_store import FaissVectorStore
from dotenv import load_dotenv
import logging
from core.utils import parse_timestamp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedMemoryLoader:
    """Optimized memory loader using pre-computed FAISS embeddings.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key used for generating embeddings during search. If not
        provided, the loader falls back to the ``OPENAI_API_KEY`` environment
        variable. A :class:`RuntimeError` is raised if neither source provides a
        key when embeddings are required.

    Usage
    -----
    ``loader = OptimizedMemoryLoader(api_key="sk-...")``
    ``engine = loader.create_optimized_memory_engine("memories.json", "faiss.index", "faiss.pkl")``

    If ``api_key`` is omitted and ``OPENAI_API_KEY`` is unset, calling
    :meth:`create_optimized_memory_engine` will raise ``RuntimeError``.
    """

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        
    def load_precomputed_memories(
        self,
        memory_json_path: Union[str, Path],
        faiss_index_path: Union[str, Path],
        faiss_pkl_path: Union[str, Path]
    ) -> tuple[List[Memory], FaissVectorStore]:
        """
        Load memories with pre-computed FAISS embeddings
        
        Args:
            memory_json_path: Path to ChatGPT memories JSON
            faiss_index_path: Path to FAISS .index file
            faiss_pkl_path: Path to FAISS .pkl metadata file
            
        Returns:
            Tuple of (memories_list, vector_store)
        """
        memory_json_path = Path(memory_json_path)
        faiss_index_path = Path(faiss_index_path)
        faiss_pkl_path = Path(faiss_pkl_path)

        logger.info("🚀 Starting optimized memory loading...")

        # 1. Load memory JSON data
        logger.info(f"📄 Loading memories from {memory_json_path}")
        with memory_json_path.open('r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        logger.info(f"✅ Loaded {len(memory_data)} memories from JSON")
        
        # 2. Load pre-computed FAISS index directly
        logger.info(f"🔍 Loading pre-computed FAISS index from {faiss_index_path}")

        # Create vector store with existing index
        vector_store = FaissVectorStore(
            dimension=1536,  # OpenAI ada-002 dimension
            index_path=str(faiss_index_path.with_suffix(''))  # Remove .index extension
        )
        
        # Verify FAISS index loaded correctly
        try:
            # Test the vector store
            if hasattr(vector_store.index, 'ntotal'):
                index_count = vector_store.index.ntotal
                logger.info(f"✅ FAISS index loaded with {index_count} vectors")
            else:
                logger.info("✅ FAISS index loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading FAISS index: {e}")
            return [], None
        
        # 3. Convert JSON data to Memory objects (without embeddings)
        logger.info("🔄 Converting JSON data to Memory objects...")
        memories = []
        
        for i, mem_data in enumerate(memory_data):
            try:
                # Parse timestamp
                timestamp = parse_timestamp(mem_data.get('timestamp'))

                # Create Memory object without embedding (will use FAISS index)
                memory = Memory(
                    content=mem_data.get('content', ''),
                    embedding=None,  # Will be retrieved from FAISS when needed
                    metadata=mem_data.get('metadata', {}),
                    timestamp=timestamp,
                    relevance_score=0.0,
                    # Enhanced fields from ChatGPT import
                    role=mem_data.get('role', 'user'),
                    thread_id=mem_data.get('thread_id'),
                    title=mem_data.get('title'),
                    type=mem_data.get('type', 'history'),
                    importance=mem_data.get('importance', 1.0)
                )
                
                memories.append(memory)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"📊 Processed {i + 1:,} memories...")
                    
            except Exception as e:
                logger.error(f"❌ Error processing memory {i}: {e}")
                continue
        
        logger.info(f"✅ Successfully created {len(memories)} Memory objects")
        
        # 4. Verify alignment between memories and FAISS index
        if hasattr(vector_store.index, 'ntotal') and vector_store.index.ntotal > 0:
            faiss_count = vector_store.index.ntotal
            memory_count = len(memories)
            
            if faiss_count != memory_count:
                logger.warning(f"⚠️  Mismatch: {memory_count} memories but {faiss_count} FAISS vectors")
                logger.info("This is normal if some memories were filtered during import")
            else:
                logger.info(f"✅ Perfect alignment: {memory_count} memories = {faiss_count} FAISS vectors")
        
        return memories, vector_store
    
    def create_optimized_memory_engine(
        self,
        memory_json_path: Union[str, Path],
        faiss_index_path: Union[str, Path],
        faiss_pkl_path: Union[str, Path]
    ):
        """
        Create a memory engine with optimized loading using pre-computed embeddings
        """
        from core.memory_engine import MemoryEngine
        
        logger.info("🏗️  Creating optimized MemoryEngine...")
        
        # Load memories and vector store
        memories, vector_store = self.load_precomputed_memories(
            memory_json_path, faiss_index_path, faiss_pkl_path
        )
        
        if not memories or not vector_store:
            logger.error("❌ Failed to load memories or vector store")
            return None
        
        # Initialize embeddings provider (required for querying)
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key is required for embeddings. Provide an api_key or set OPENAI_API_KEY."
            )

        embeddings_provider = OpenAIEmbeddings(self.api_key)
        logger.info("✅ OpenAI embeddings provider ready for new queries")
        
        # Create memory engine with pre-loaded data
        memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embeddings_provider,
            persist_path=None  # Don't auto-save to prevent overwriting
        )
        
        # Manually set the memories (bypass the loading process)
        memory_engine.memories = memories
        logger.info(f"✅ MemoryEngine created with {len(memories)} pre-loaded memories")
        
        return memory_engine
    
    def test_search_performance(self, memory_engine, test_queries: List[str] = None):
        """Test search performance with the optimized loader"""
        if not test_queries:
            test_queries = [
                "python programming",
                "LangChain OpenAI",
                "vector database",
                "machine learning AI",
                "ChatGPT conversation"
            ]
        
        logger.info("🔍 Testing search performance...")
        
        for query in test_queries:
            try:
                start_time = datetime.now()
                results = memory_engine.search_memories(query, k=3)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                logger.info(f"✅ '{query}': {len(results)} results in {duration:.3f}s")
                
                # Show top result
                if results:
                    top_result = results[0]
                    logger.info(f"   📝 Top: {top_result.content[:100]}... (score: {top_result.relevance_score:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Search failed for '{query}': {e}")

def main():
    """Main function to test the optimized loader"""
    if len(sys.argv) < 4:
        print("Usage: python optimized_memory_loader.py <memory.json> <faiss.index> <faiss.pkl>")
        print("Example: python optimized_memory_loader.py data/chatgpt_memories.json data/faiss_chatgpt_index.index data/faiss_chatgpt_index.pkl")
        sys.exit(1)
    
    memory_json = Path(sys.argv[1])
    faiss_index = Path(sys.argv[2])
    faiss_pkl = Path(sys.argv[3])

    # Verify files exist
    for file_path in [memory_json, faiss_index, faiss_pkl]:
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            sys.exit(1)
    
    # Create optimized loader
    loader = OptimizedMemoryLoader()
    
    try:
        # Create memory engine
        memory_engine = loader.create_optimized_memory_engine(
            memory_json, faiss_index, faiss_pkl
        )
    except RuntimeError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)

    if memory_engine:
        # Test search performance
        loader.test_search_performance(memory_engine)

        logger.info("🎉 Optimized memory loading completed successfully!")
        logger.info(f"💾 {len(memory_engine.memories)} memories ready for instant search")

        return memory_engine
    else:
        logger.error("❌ Failed to create optimized memory engine")
        sys.exit(1)

if __name__ == "__main__":
    main()
