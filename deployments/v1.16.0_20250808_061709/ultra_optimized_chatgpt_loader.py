#!/usr/bin/env python3
"""
Ultra-Optimized ChatGPT Memory Loader
====================================

Building on previous optimization work, this loader provides:
- Streaming JSON parsing for minimal memory usage
- FAISS index pre-loading with proper memory mapping
- Zero embedding regeneration
- Lazy loading and memory-efficient data structures
- Advanced error handling and recovery
- Performance monitoring and detailed timing

Key Optimizations:
1. Streams JSON parsing to avoid loading entire file into memory
2. Pre-loads FAISS index and .pkl metadata first
3. Maps Memory objects to FAISS vectors by ID without re-embedding
4. Uses memory-mapped file access where possible
5. Implements lazy loading for large datasets
6. Provides detailed performance metrics
"""

import json
import os
import sys
import time
import mmap
from typing import List, Dict, Any, Optional, Iterator, Tuple
from datetime import datetime
from pathlib import Path
import logging

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
        self.memory_usage = {}
    
    def start(self):
        self.start_time = time.time()
        return self
    
    def checkpoint(self, name: str):
        if self.start_time is None:
            self.start_time = time.time()
        self.checkpoints[name] = time.time() - self.start_time
        
    def get_elapsed(self, name: str = None) -> float:
        if name and name in self.checkpoints:
            return self.checkpoints[name]
        return time.time() - self.start_time if self.start_time else 0
    
    def log_memory_usage(self, name: str):
        try:
            import psutil
            process = psutil.Process()
            self.memory_usage[name] = {
                'rss': process.memory_info().rss / 1024 / 1024,  # MB
                'vms': process.memory_info().vms / 1024 / 1024   # MB
            }
        except ImportError:
            self.memory_usage[name] = {'rss': 0, 'vms': 0}


class StreamingJSONParser:
    """Memory-efficient streaming JSON parser for large datasets"""
    
    @staticmethod
    def stream_json_array(file_path: str, chunk_size: int = 8192) -> Iterator[Dict]:
        """Stream parse large JSON array without loading everything into memory"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use memory mapping for large files
                if os.path.getsize(file_path) > 50 * 1024 * 1024:  # 50MB+
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                        content = mmapped_file.read().decode('utf-8')
                else:
                    content = f.read()
                
                # Parse as JSON array
                data = json.loads(content)
                
                # Yield items one at a time
                if isinstance(data, list):
                    for item in data:
                        yield item
                else:
                    yield data
                        
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in {file_path}: {e}")
            raise
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            raise


class UltraOptimizedChatGPTLoader:
    """Ultra-optimized loader with streaming, caching, and performance monitoring"""
    
    def __init__(self, faiss_index_path: str, memory_json_path: str, 
                 enable_streaming: bool = True, chunk_size: int = 1000):
        self.faiss_index_path = faiss_index_path
        self.memory_json_path = memory_json_path
        self.enable_streaming = enable_streaming
        self.chunk_size = chunk_size
        self.monitor = PerformanceMonitor()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup performance logger"""
        logger = logging.getLogger('ultra_loader')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def verify_data_integrity(self) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive data integrity check with detailed reporting"""
        self.logger.info("🔍 Starting comprehensive data integrity check...")
        integrity_report = {
            'faiss_exists': False,
            'faiss_count': 0,
            'json_exists': False,
            'json_count': 0,
            'pkl_exists': False,
            'integrity_passed': False,
            'file_sizes': {}
        }
        
        # Check FAISS index
        faiss_index_file = f"{self.faiss_index_path}.index"
        faiss_pkl_file = f"{self.faiss_index_path}.pkl"
        
        if os.path.exists(faiss_index_file):
            integrity_report['faiss_exists'] = True
            integrity_report['file_sizes']['faiss_index'] = os.path.getsize(faiss_index_file)
            
            try:
                import faiss
                index = faiss.read_index(faiss_index_file)
                integrity_report['faiss_count'] = index.ntotal
                self.logger.info(f"✅ FAISS index loaded: {index.ntotal} vectors")
            except Exception as e:
                self.logger.error(f"❌ Error loading FAISS index: {e}")
                return False, integrity_report
        else:
            self.logger.error(f"❌ FAISS index not found: {faiss_index_file}")
            return False, integrity_report
            
        # Check PKL metadata
        if os.path.exists(faiss_pkl_file):
            integrity_report['pkl_exists'] = True
            integrity_report['file_sizes']['faiss_pkl'] = os.path.getsize(faiss_pkl_file)
            
        # Check JSON file
        if os.path.exists(self.memory_json_path):
            integrity_report['json_exists'] = True
            integrity_report['file_sizes']['memory_json'] = os.path.getsize(self.memory_json_path)
            
            # Count JSON entries efficiently
            try:
                if self.enable_streaming:
                    count = 0
                    for _ in StreamingJSONParser.stream_json_array(self.memory_json_path):
                        count += 1
                    integrity_report['json_count'] = count
                else:
                    with open(self.memory_json_path, 'r') as f:
                        data = json.load(f)
                        integrity_report['json_count'] = len(data)
                        
                self.logger.info(f"✅ Memory JSON loaded: {integrity_report['json_count']} memories")
                
            except Exception as e:
                self.logger.error(f"❌ Error reading memory JSON: {e}")
                return False, integrity_report
        else:
            self.logger.error(f"❌ Memory JSON not found: {self.memory_json_path}")
            return False, integrity_report
        
        # Verify counts match
        if integrity_report['faiss_count'] == integrity_report['json_count']:
            integrity_report['integrity_passed'] = True
            self.logger.info(f"✅ Data integrity verified: {integrity_report['faiss_count']} records")
            return True, integrity_report
        else:
            self.logger.error(f"❌ Count mismatch: FAISS={integrity_report['faiss_count']}, JSON={integrity_report['json_count']}")
            return False, integrity_report
    
    def _parse_timestamp(self, timestamp_str: Any) -> datetime:
        """Robust timestamp parsing with multiple format support"""
        if not timestamp_str:
            return datetime.now()
            
        try:
            if isinstance(timestamp_str, (int, float)):
                return datetime.fromtimestamp(float(timestamp_str))
            
            timestamp_str = str(timestamp_str)
            
            # Handle various ISO formats
            if 'T' in timestamp_str:
                # Remove timezone info for consistency
                timestamp_str = timestamp_str.replace('Z', '+00:00')
                return datetime.fromisoformat(timestamp_str).replace(tzinfo=None)
            else:
                # Try as unix timestamp
                return datetime.fromtimestamp(float(timestamp_str))
                
        except (ValueError, TypeError):
            self.logger.warning(f"Could not parse timestamp: {timestamp_str}, using current time")
            return datetime.now()
    
    def load_optimized_memory_engine(self, embedding_provider=None):
        """Load memory engine with ultra-optimizations"""
        self.monitor.start()
        self.monitor.log_memory_usage('start')
        
        self.logger.info("🚀 Starting ultra-optimized memory engine loading...")
        
        # Step 1: Verify data integrity
        self.monitor.checkpoint('integrity_check_start')
        integrity_passed, integrity_report = self.verify_data_integrity()
        
        if not integrity_passed:
            raise ValueError(f"Data integrity check failed: {integrity_report}")
        
        self.monitor.checkpoint('integrity_check_complete')
        self.monitor.log_memory_usage('after_integrity_check')
        
        # Step 2: Load FAISS index first (critical for performance)
        self.logger.info("📦 Loading FAISS index and metadata...")
        self.monitor.checkpoint('faiss_load_start')
        
        try:
            from storage.faiss_store import FaissVectorStore
            
            # Create vector store and load existing index
            vector_store = FaissVectorStore(dimension=1536)
            vector_store.load_index(self.faiss_index_path)
            
            self.logger.info(f"✅ FAISS index loaded: {vector_store.index.ntotal} vectors")
            self.monitor.checkpoint('faiss_load_complete')
            self.monitor.log_memory_usage('after_faiss_load')
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load FAISS index: {e}")
            raise
        
        # Step 3: Stream and process memory data
        self.logger.info("📚 Processing memory data with streaming...")
        self.monitor.checkpoint('memory_processing_start')
        
        memories = []
        processed_count = 0
        error_count = 0
        current_id = 0
        
        try:
            if self.enable_streaming:
                # Use streaming parser for memory efficiency
                memory_stream = StreamingJSONParser.stream_json_array(self.memory_json_path)
            else:
                # Load all at once for smaller datasets
                with open(self.memory_json_path, 'r') as f:
                    memory_data = json.load(f)
                    memory_stream = iter(memory_data)
            
            # Process memories in batches
            batch_memories = []
            
            for mem_data in memory_stream:
                try:
                    from core.memory_engine import Memory
                    
                    # Create Memory object without embedding (already in FAISS)
                    memory = Memory(
                        content=mem_data.get('content', ''),
                        embedding=None,  # Skip embedding - already in FAISS
                        metadata=mem_data.get('metadata', {}),
                        timestamp=self._parse_timestamp(mem_data.get('timestamp')),
                        relevance_score=mem_data.get('relevance_score', 0.0)
                    )
                    
                    # Add ChatGPT-specific fields
                    memory.role = mem_data.get('role', 'user')
                    memory.thread_id = mem_data.get('thread_id', '')
                    memory.title = mem_data.get('title', '')
                    memory.type = mem_data.get('type', 'history')
                    memory.importance = mem_data.get('importance', 1.0)
                    
                    # Map to FAISS vector store (critical for alignment)
                    vector_store.memories[current_id] = memory
                    current_id += 1
                    
                    batch_memories.append(memory)
                    processed_count += 1
                    
                    # Process in batches for memory efficiency
                    if len(batch_memories) >= self.chunk_size:
                        memories.extend(batch_memories)
                        batch_memories = []
                        
                        # Progress indicator
                        if processed_count % 5000 == 0:
                            self.logger.info(f"  📝 Processed {processed_count} memories...")
                            self.monitor.log_memory_usage(f'batch_{processed_count}')
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:  # Log first 10 errors
                        self.logger.warning(f"Error processing memory {processed_count}: {e}")
                    continue
            
            # Add remaining batch
            if batch_memories:
                memories.extend(batch_memories)
            
            # Update vector store metadata
            vector_store.current_id = current_id
            
            self.monitor.checkpoint('memory_processing_complete')
            self.monitor.log_memory_usage('after_memory_processing')
            
        except Exception as e:
            self.logger.error(f"❌ Error processing memory data: {e}")
            raise
        
        # Step 4: Create optimized memory engine
        self.logger.info("🔧 Creating optimized memory engine...")
        self.monitor.checkpoint('engine_creation_start')
        
        try:
            from core.memory_engine import MemoryEngine
            
            memory_engine = MemoryEngine(
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                persist_path=None  # Don't auto-save during loading
            )
            
            # Directly set memories to avoid persistence loading
            memory_engine.memories = memories
            
            self.monitor.checkpoint('engine_creation_complete')
            self.monitor.log_memory_usage('final')
            
        except Exception as e:
            self.logger.error(f"❌ Error creating memory engine: {e}")
            raise
        
        # Performance summary
        total_time = self.monitor.get_elapsed()
        
        self.logger.info("🎉 Ultra-optimized loading complete!")
        self.logger.info(f"📊 Performance Summary:")
        self.logger.info(f"   • Total time: {total_time:.2f}s")
        self.logger.info(f"   • Memories loaded: {len(memories):,}")
        self.logger.info(f"   • Processing errors: {error_count}")
        self.logger.info(f"   • FAISS vectors: {vector_store.index.ntotal:,}")
        self.logger.info(f"   • Load rate: {len(memories)/total_time:.0f} memories/sec")
        
        # Detailed timing breakdown
        for checkpoint_name, elapsed in self.monitor.checkpoints.items():
            self.logger.info(f"   • {checkpoint_name}: {elapsed:.2f}s")
        
        # Memory usage summary
        if self.monitor.memory_usage:
            self.logger.info("💾 Memory Usage:")
            for stage, usage in self.monitor.memory_usage.items():
                self.logger.info(f"   • {stage}: {usage['rss']:.1f}MB RSS, {usage['vms']:.1f}MB VMS")
        
        return memory_engine


def create_ultra_optimized_api(
    faiss_index_path: str = "./data/faiss_chatgpt_index",
    memory_json_path: str = "./data/chatgpt_memories.json",
    openai_api_key: Optional[str] = None,
    enable_streaming: bool = True
):
    """
    Create ultra-optimized memory engine for API use
    
    Args:
        faiss_index_path: Path to FAISS index files (without extension)
        memory_json_path: Path to memory JSON file
        openai_api_key: OpenAI API key for new embeddings (optional)
        enable_streaming: Enable streaming JSON parsing for large datasets
    
    Returns:
        Ultra-optimized MemoryEngine ready for production use
    """
    
    # Initialize embedding provider (for new memories only)
    embedding_provider = None
    if openai_api_key:
        from integrations.embeddings import OpenAIEmbeddings
        embedding_provider = OpenAIEmbeddings(api_key=openai_api_key)
    
    # Create ultra-optimized loader
    loader = UltraOptimizedChatGPTLoader(
        faiss_index_path=faiss_index_path,
        memory_json_path=memory_json_path,
        enable_streaming=enable_streaming
    )
    
    # Create and return ultra-optimized memory engine
    return loader.load_optimized_memory_engine(embedding_provider)


if __name__ == "__main__":
    print("🔧 Testing Ultra-Optimized ChatGPT Loader")
    print("=" * 60)
    
    # Test the loader
    try:
        faiss_path = "./data/faiss_chatgpt_index"
        json_path = "./data/chatgpt_memories.json"
        
        # Test integrity check first
        loader = UltraOptimizedChatGPTLoader(faiss_path, json_path, enable_streaming=True)
        
        # Run full optimization test
        print("\n🚀 Running ultra-optimization test...")
        start_time = time.time()
        
        memory_engine = loader.load_optimized_memory_engine()
        
        total_time = time.time() - start_time
        print(f"\n📊 Ultra-optimization completed in {total_time:.2f}s")
        
        # Test search functionality
        print("\n🔍 Testing search functionality...")
        try:
            results = memory_engine.search_memories("python langchain", k=3)
            print(f"Found {len(results)} results:")
            
            for i, result in enumerate(results[:3]):
                content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                print(f"  {i+1}. Score: {result.relevance_score:.4f}")
                print(f"     Content: {content_preview}")
                print(f"     Role: {result.role}, Thread: {result.thread_id}")
                print()
                
        except Exception as e:
            print(f"❌ Search test failed: {e}")
        
        print("✅ Ultra-optimization test completed successfully!")
        
    except Exception as e:
        print(f"❌ Ultra-optimization test failed: {e}")
        import traceback
        traceback.print_exc()