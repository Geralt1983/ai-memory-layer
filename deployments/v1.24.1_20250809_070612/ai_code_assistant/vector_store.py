#!/usr/bin/env python3
"""
Vector Store
============

Simple but efficient vector storage and retrieval using FAISS.
Stores embeddings with metadata for semantic search and retrieval.
"""

import os
import json
import pickle
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import numpy as np
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """Represents a stored memory with embedding"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: str
    similarity: Optional[float] = None  # Set during search

class VectorStore:
    """FAISS-based vector storage with SQLite metadata"""
    
    def __init__(self, data_dir: Path, dimension: int = 1536):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dimension = dimension
        self.index_path = self.data_dir / "faiss_index.bin"
        self.metadata_db = self.data_dir / "metadata.db"
        self.processed_files = self.data_dir / "processed_files.txt"
        
        # Initialize FAISS index
        if FAISS_AVAILABLE:
            self._init_faiss_index()
        else:
            logger.warning("FAISS not available, falling back to simple storage")
            self.index = None
        
        # Initialize SQLite database
        self._init_metadata_db()
        
        # Load processed files set
        self._load_processed_files()
        
        logger.info(f"Vector store initialized with dimension {dimension}")
    
    def _init_faiss_index(self):
        """Initialize or load FAISS index"""
        try:
            if self.index_path.exists():
                # Load existing index
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index (L2 distance, flat index for simplicity)
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = None
    
    def _init_metadata_db(self):
        """Initialize SQLite database for metadata"""
        try:
            conn = sqlite3.connect(self.metadata_db)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT,
                    vector_index INTEGER
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stats (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            conn.commit()
            conn.close()
            logger.info("Metadata database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize metadata database: {e}")
    
    def _load_processed_files(self):
        """Load set of processed file hashes"""
        self.processed_file_hashes = set()
        if self.processed_files.exists():
            try:
                with open(self.processed_files, 'r') as f:
                    self.processed_file_hashes = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(self.processed_file_hashes)} processed file hashes")
            except Exception as e:
                logger.warning(f"Failed to load processed files: {e}")
    
    def _save_processed_files(self):
        """Save processed file hashes"""
        try:
            with open(self.processed_files, 'w') as f:
                for file_hash in sorted(self.processed_file_hashes):
                    f.write(f"{file_hash}\n")
        except Exception as e:
            logger.warning(f"Failed to save processed files: {e}")
    
    def is_processed(self, file_hash: str) -> bool:
        """Check if a file hash has been processed"""
        return file_hash in self.processed_file_hashes
    
    def mark_processed(self, file_hash: str):
        """Mark a file hash as processed"""
        self.processed_file_hashes.add(file_hash)
        self._save_processed_files()
    
    def add_memory(self, content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """Add a memory to the vector store"""
        import hashlib
        from datetime import datetime
        
        # Generate unique ID
        memory_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        try:
            # Add to FAISS index if available
            vector_index = -1
            if self.index is not None and FAISS_AVAILABLE:
                embedding_array = np.array([embedding], dtype=np.float32)
                vector_index = self.index.ntotal
                self.index.add(embedding_array)
                
                # Save updated index
                faiss.write_index(self.index, str(self.index_path))
            
            # Store metadata in SQLite
            conn = sqlite3.connect(self.metadata_db)
            conn.execute('''
                INSERT OR REPLACE INTO memories (id, content, metadata, timestamp, vector_index)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                memory_id,
                content,
                json.dumps(metadata),
                datetime.now().isoformat(),
                vector_index
            ))
            conn.commit()
            conn.close()
            
            logger.debug(f"Added memory {memory_id} with {len(content)} characters")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
    
    def search(self, query_embedding: List[float], max_results: int = 5) -> List[Memory]:
        """Search for similar memories"""
        if not self.index or not FAISS_AVAILABLE:
            # Fallback to text-based search
            return self._fallback_search("", max_results)
        
        try:
            # Search FAISS index
            query_array = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_array, max_results)
            
            # Get metadata for results
            conn = sqlite3.connect(self.metadata_db)
            memories = []
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # No more results
                    break
                
                cursor = conn.execute('''
                    SELECT id, content, metadata, timestamp FROM memories 
                    WHERE vector_index = ?
                ''', (int(idx),))
                
                row = cursor.fetchone()
                if row:
                    memory_id, content, metadata_json, timestamp = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    # Convert L2 distance to similarity score (0-1, higher is better)
                    similarity = float(1.0 / (1.0 + distance))
                    
                    memory = Memory(
                        id=memory_id,
                        content=content,
                        embedding=query_embedding,  # We don't store the original embedding
                        metadata=metadata,
                        timestamp=timestamp,
                        similarity=similarity
                    )
                    memories.append(memory)
            
            conn.close()
            
            logger.debug(f"Found {len(memories)} similar memories")
            return memories
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _fallback_search(self, query: str, max_results: int = 5) -> List[Memory]:
        """Fallback text-based search when FAISS is not available"""
        try:
            conn = sqlite3.connect(self.metadata_db)
            
            # Simple text search in content
            cursor = conn.execute('''
                SELECT id, content, metadata, timestamp FROM memories 
                WHERE content LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (f'%{query}%', max_results))
            
            memories = []
            for row in cursor:
                memory_id, content, metadata_json, timestamp = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                memory = Memory(
                    id=memory_id,
                    content=content,
                    embedding=[],
                    metadata=metadata,
                    timestamp=timestamp,
                    similarity=0.5  # Default similarity
                )
                memories.append(memory)
            
            conn.close()
            return memories
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def get_recent_memories(self, limit: int = 10) -> List[Memory]:
        """Get recent memories"""
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.execute('''
                SELECT id, content, metadata, timestamp FROM memories 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            memories = []
            for row in cursor:
                memory_id, content, metadata_json, timestamp = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                memory = Memory(
                    id=memory_id,
                    content=content,
                    embedding=[],
                    metadata=metadata,
                    timestamp=timestamp
                )
                memories.append(memory)
            
            conn.close()
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID"""
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.execute('''
                SELECT id, content, metadata, timestamp FROM memories 
                WHERE id = ?
            ''', (memory_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                memory_id, content, metadata_json, timestamp = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                return Memory(
                    id=memory_id,
                    content=content,
                    embedding=[],
                    metadata=metadata,
                    timestamp=timestamp
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory (note: doesn't remove from FAISS index)"""
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if deleted:
                logger.info(f"Deleted memory {memory_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            conn = sqlite3.connect(self.metadata_db)
            
            # Total memories
            cursor = conn.execute('SELECT COUNT(*) FROM memories')
            total_memories = cursor.fetchone()[0]
            
            # Commit memories
            cursor = conn.execute('''
                SELECT COUNT(*) FROM memories 
                WHERE json_extract(metadata, '$.type') = 'commit'
            ''')
            commit_memories = cursor.fetchone()[0]
            
            # Recent memories (last 7 days)
            cursor = conn.execute('''
                SELECT COUNT(*) FROM memories 
                WHERE datetime(timestamp) > datetime('now', '-7 days')
            ''')
            recent_memories = cursor.fetchone()[0]
            
            conn.close()
            
            stats = {
                "total_memories": total_memories,
                "commit_memories": commit_memories,
                "recent_memories": recent_memories,
                "processed_files": len(self.processed_file_hashes),
                "faiss_available": FAISS_AVAILABLE,
                "index_size": self.index.ntotal if self.index else 0,
                "dimension": self.dimension
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def clear_all(self):
        """Clear all memories (use with caution!)"""
        try:
            # Clear FAISS index
            if self.index:
                self.index.reset()
                if self.index_path.exists():
                    self.index_path.unlink()
            
            # Clear SQLite database
            conn = sqlite3.connect(self.metadata_db)
            conn.execute('DELETE FROM memories')
            conn.commit()
            conn.close()
            
            # Clear processed files
            self.processed_file_hashes.clear()
            if self.processed_files.exists():
                self.processed_files.unlink()
            
            logger.info("Cleared all memories")
            
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            raise