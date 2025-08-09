#!/usr/bin/env python3
"""
Optimized ChatGPT Conversation Importer with Batching and Resume
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

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

class OptimizedImporter:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.progress_file = "./data/import_progress.json"
        self.memory_engine = None
        self.setup_memory_engine()
        
    def setup_memory_engine(self):
        """Initialize memory engine components"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        # Set up components
        embeddings = OpenAIEmbeddings(api_key)
        vector_store = FaissVectorStore(
            dimension=1536,
            index_path="./data/faiss_chatgpt_index"
        )
        
        self.memory_engine = MemoryEngine(
            vector_store=vector_store,
            embedding_provider=embeddings,
            persist_path="./data/chatgpt_memories.json"
        )
        
    def save_progress(self, processed_count: int, total_count: int, errors: int):
        """Save import progress"""
        os.makedirs("./data", exist_ok=True)
        progress = {
            "processed_count": processed_count,
            "total_count": total_count,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "completion_percentage": (processed_count / total_count) * 100
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self) -> Dict[str, Any]:
        """Load previous import progress"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"processed_count": 0, "total_count": 0, "errors": 0}
    
    def import_conversations(self, file_path: str):
        """Import conversations with progress tracking and batching"""
        # Load messages
        logger.info(f"Loading messages from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        
        total_messages = len(messages)
        logger.info(f"Total messages to process: {total_messages}")
        
        # Load previous progress
        progress = self.load_progress()
        start_index = progress.get("processed_count", 0)
        errors = progress.get("errors", 0)
        
        if start_index > 0:
            logger.info(f"Resuming from message {start_index} ({start_index/total_messages*100:.1f}%)")
        
        # Process messages in batches
        processed = start_index
        batch_start_time = time.time()
        
        for i in range(start_index, total_messages, self.batch_size):
            batch = messages[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_messages + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} (messages {i+1}-{min(i+self.batch_size, total_messages)})")
            
            # Process batch with error handling
            batch_errors = 0
            for msg_idx, msg in enumerate(batch):
                try:
                    # Add memory with rate limiting consideration
                    self.memory_engine.add_memory(
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
                    processed += 1
                    
                    # Add small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing message {i + msg_idx + 1}: {str(e)}")
                    batch_errors += 1
                    errors += 1
                    
                    # Stop if too many errors
                    if batch_errors > 10:
                        logger.error("Too many errors in batch, stopping")
                        break
            
            # Save progress after each batch
            self.save_progress(processed, total_messages, errors)
            
            # Show progress
            elapsed = time.time() - batch_start_time
            rate = self.batch_size / elapsed if elapsed > 0 else 0
            completion = (processed / total_messages) * 100
            
            logger.info(f"Batch completed: {processed}/{total_messages} ({completion:.1f}%) - Rate: {rate:.1f} msg/sec - Errors: {errors}")
            
            # Reset timer for next batch
            batch_start_time = time.time()
            
            # Break if too many total errors
            if errors > 100:
                logger.error("Too many total errors, stopping import")
                break
        
        # Final save and summary
        logger.info(f"\nImport completed!")
        logger.info(f"Total processed: {processed}/{total_messages}")
        logger.info(f"Success rate: {((processed-errors)/processed)*100:.1f}%")
        logger.info(f"Total errors: {errors}")
        
        # Save final vector store
        if hasattr(self.memory_engine.vector_store, 'save_index'):
            self.memory_engine.vector_store.save_index("./data/faiss_chatgpt_index")
            logger.info("Vector store saved successfully")
        
        # Test search if successful
        if processed > 100:
            self.test_search()
    
    def test_search(self):
        """Test the imported memories with sample searches"""
        logger.info("\nTesting search functionality...")
        test_queries = [
            "AI memory layer",
            "Python code",
            "OpenAI API",
            "vector database",
            "machine learning"
        ]
        
        for query in test_queries:
            try:
                results = self.memory_engine.search_memories(query, k=3)
                logger.info(f"'{query}': {len(results)} results")
                if results:
                    logger.info(f"  Top: {results[0].content[:80]}...")
            except Exception as e:
                logger.error(f"Search error for '{query}': {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python import_conversations_optimized.py <path_to_processed.json>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Create importer and start
    importer = OptimizedImporter(batch_size=50)  # Smaller batches for better progress tracking
    importer.import_conversations(file_path)

if __name__ == "__main__":
    main()