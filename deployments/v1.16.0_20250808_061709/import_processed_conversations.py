#!/usr/bin/env python3
"""
Import Processed Conversations with Embeddings
Takes the processed JSON from chatgpt_importer_no_embed.py and generates embeddings
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from core.memory_engine import MemoryEngine
from integrations.embeddings import OpenAIEmbeddings
from storage.faiss_store import FaissVectorStore
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def import_processed_conversations(file_path: str, batch_size: int = 50):
    """Import processed conversations and generate embeddings"""
    
    # Load processed messages
    logger.info(f"Loading processed conversations from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    logger.info(f"Loaded {len(messages)} messages")
    
    # Initialize components
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    # Set up memory engine with vector store
    embeddings = OpenAIEmbeddings(api_key)
    vector_store = FaissVectorStore(
        dimension=1536,  # OpenAI ada-002 embedding dimension
        index_path="./data/faiss_index"
    )
    
    memory_engine = MemoryEngine(
        vector_store=vector_store,
        embedding_provider=embeddings,
        persist_path="./data/chatgpt_memories.json"
    )
    
    # Process messages in batches
    total_imported = 0
    errors = 0
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(messages) + batch_size - 1)//batch_size}")
        
        for msg in batch:
            try:
                # Add to memory engine with enhanced metadata
                memory = memory_engine.add_memory(
                    content=msg.get('text', ''),
                    role=msg.get('role', 'user'),
                    thread_id=msg.get('thread_id'),
                    title=msg.get('title'),
                    type=msg.get('type', 'history'),
                    importance=msg.get('importance', 1.0),
                    metadata={
                        'source': 'chatgpt_import',
                        'original_timestamp': msg.get('timestamp'),
                        'import_timestamp': datetime.now().isoformat()
                    }
                )
                total_imported += 1
                
                if total_imported % 100 == 0:
                    logger.info(f"Imported {total_imported} messages with embeddings")
                    
            except Exception as e:
                logger.error(f"Error importing message: {str(e)}")
                errors += 1
                if errors > 10:
                    logger.error("Too many errors, stopping import")
                    break
    
    # Save the vector store
    if hasattr(vector_store, 'save_index'):
        vector_store.save_index("./data/faiss_index")
        logger.info("Saved FAISS index")
    
    logger.info(f"\nImport completed!")
    logger.info(f"Total messages imported: {total_imported}")
    logger.info(f"Errors: {errors}")
    
    # Test search functionality
    if total_imported > 0:
        logger.info("\nTesting search functionality...")
        test_queries = [
            "AI memory layer",
            "LangChain",
            "OpenAI API",
            "vector storage",
            "conversation context"
        ]
        
        for query in test_queries:
            results = memory_engine.search_memories(query, k=3)
            logger.info(f"\nSearch for '{query}' returned {len(results)} results")
            if results:
                logger.info(f"Top result: {results[0].content[:100]}...")


def main():
    """Main import function"""
    if len(sys.argv) < 2:
        print("Usage: python import_processed_conversations.py <path_to_processed.json>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    import_processed_conversations(file_path)


if __name__ == "__main__":
    main()