from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import os

# Try to import numpy, fall back to mock if not available
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    print("Info: Running in numpy-free mode for compatibility")
    NUMPY_AVAILABLE = False

    # Mock numpy for type compatibility
    class MockArray:
        def __init__(self, data=None):
            self.data = data if data is not None else []

        def tolist(self):
            return self.data if isinstance(self.data, list) else [self.data]

        def __str__(self):
            return f"MockArray({self.data})"

    class MockNumpy:
        ndarray = MockArray

        @staticmethod
        def array(data):
            return MockArray(data)

    np = MockNumpy()
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from .logging_config import get_logger, log_memory_operation, monitor_performance


@dataclass
class Memory:
    content: str  # The 'text' field from ChatGPT format
    embedding: Optional[Union[np.ndarray, list]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    
    # New fields for enhanced memory system
    role: str = "user"  # "user" or "assistant"
    thread_id: Optional[str] = None
    title: Optional[str] = None
    type: str = "history"  # "history", "identity", "correction", "summary"
    importance: float = 1.0  # 0.0 to 1.0 for retrieval weighting

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score,
            "role": self.role,
            "thread_id": self.thread_id,
            "title": self.title,
            "type": self.type,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        # Handle both old and new formats for backward compatibility
        return cls(
            content=data.get("content", data.get("text", "")),  # Support 'text' field
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now()),
            relevance_score=data.get("relevance_score", 0.0),
            role=data.get("role", "user"),
            thread_id=data.get("thread_id"),
            title=data.get("title"),
            type=data.get("type", "history"),
            importance=data.get("importance", 1.0),
        )


class VectorStore(ABC):
    @abstractmethod
    def add_memory(self, memory: Memory) -> str:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Memory]:
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        pass


class MemoryEngine:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_provider: Optional["EmbeddingProvider"] = None,
        persist_path: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.persist_path = persist_path
        self.memories: List[Memory] = []
        self.logger = get_logger("memory_engine")

        self.logger.info(
            "Initializing MemoryEngine",
            extra={
                "vector_store_type": (
                    type(vector_store).__name__ if vector_store else None
                ),
                "embedding_provider_type": (
                    type(embedding_provider).__name__ if embedding_provider else None
                ),
                "persist_path": persist_path,
            },
        )

        # Load existing memories if persist path is provided
        if self.persist_path:
            self.load_memories()

    @monitor_performance("add_memory")
    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        role: str = "user",
        thread_id: Optional[str] = None,
        title: Optional[str] = None,
        type: str = "history",
        importance: float = 1.0,
    ) -> Memory:
        self.logger.debug(
            "Adding new memory",
            extra={
                "content_length": len(content),
                "metadata_keys": list((metadata or {}).keys()),
                "type": type,
                "importance": importance,
                "thread_id": thread_id,
            },
        )

        memory = Memory(
            content=content,
            metadata=metadata or {},
            role=role,
            thread_id=thread_id,
            title=title,
            type=type,
            importance=importance,
        )

        # Generate embedding if embedding provider is available
        if self.embedding_provider:
            self.logger.debug("Generating embedding for memory")
            memory.embedding = self.embedding_provider.embed_text(content)

        self.memories.append(memory)

        if self.vector_store and memory.embedding is not None:
            self.logger.debug("Adding memory to vector store")
            self.vector_store.add_memory(memory)

        # Save to persistent storage
        if self.persist_path:
            self.logger.debug("Persisting memories to storage")
            self.save_memories()

        log_memory_operation(
            "add", content_length=len(content), total_memories=len(self.memories)
        )
        self.logger.info(
            "Memory added successfully",
            extra={
                "memory_index": len(self.memories) - 1,
                "total_memories": len(self.memories),
            },
        )

        return memory
    
    def add_identity_correction(self, correction: str, context: str = "") -> Memory:
        """Add a high-priority identity/correction memory that will always be included"""
        correction_content = f"IDENTITY CORRECTION: {correction}"
        if context:
            correction_content += f" Context: {context}"
        
        metadata = {
            "type": "correction",
            "priority": "high",
            "category": "identity",
            "timestamp": datetime.now().isoformat()
        }
        
        # Remove any existing similar corrections to avoid duplicates
        self.memories = [m for m in self.memories 
                        if not (m.metadata.get("type") == "correction" and 
                               correction.lower() in m.content.lower())]
        
        self.logger.info(f"Adding identity correction: {correction}")
        return self.add_memory(correction_content, metadata)
    
    def get_high_priority_memories(self, limit: int = 3) -> List[Memory]:
        """Get high-priority memories (corrections, identity info) for system context"""
        high_priority = [m for m in self.memories 
                        if m.metadata.get("priority") == "high"]
        
        # Sort by recency
        high_priority.sort(key=lambda m: m.timestamp, reverse=True)
        return high_priority[:limit]

    @monitor_performance("search_memories")
    def search_memories(self, query: str, k: int = 5, include_importance: bool = True) -> List[Memory]:
        self.logger.debug(
            "Searching memories",
            extra={
                "query_length": len(query),
                "k": k,
                "total_memories": len(self.memories),
                "include_importance": include_importance,
            },
        )

        if not self.vector_store or not self.embedding_provider:
            self.logger.warning(
                "No vector store or embedding provider, falling back to recent memories"
            )
            return self.get_recent_memories(k)

        # Generate embedding for the query
        self.logger.debug("Generating query embedding")
        query_embedding = self.embedding_provider.embed_text(query)

        # Search using the vector store - get more results for re-ranking
        search_k = k * 2 if include_importance else k
        self.logger.debug("Performing vector search", extra={"search_k": search_k})
        results = self.vector_store.search(query_embedding, search_k)

        # Apply importance weighting if enabled
        if include_importance and results:
            import math
            
            for memory in results:
                # Calculate age decay (optional, can be disabled)
                age_days = (datetime.now() - memory.timestamp).days
                age_decay = math.exp(-age_days / 30)  # 30-day half-life
                
                # Boost score based on importance and type
                importance_boost = 1 + (0.5 * memory.importance)
                
                # Extra boost for critical types
                type_boost = 1.0
                if memory.type == "correction" or memory.type == "identity":
                    type_boost = 1.5
                elif memory.type == "summary":
                    type_boost = 1.2
                
                # Apply weighted scoring
                memory.relevance_score = memory.relevance_score * importance_boost * type_boost * age_decay
                
                self.logger.debug(
                    f"Memory scoring: base={memory.relevance_score:.3f}, "
                    f"importance={memory.importance}, type={memory.type}, "
                    f"age_days={age_days}, final_score={memory.relevance_score:.3f}"
                )
            
            # Re-sort by new scores and take top k
            results.sort(key=lambda m: m.relevance_score, reverse=True)
            results = results[:k]

        log_memory_operation(
            "search", query_length=len(query), results_count=len(results)
        )
        self.logger.info(
            "Memory search completed",
            extra={"query_length": len(query), "results_count": len(results), "k": k},
        )

        return results

    def get_recent_memories(self, n: int = 10) -> List[Memory]:
        return sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:n]

    def clear_memories(self) -> None:
        memory_count = len(self.memories)
        self.logger.info("Clearing all memories", extra={"memory_count": memory_count})

        self.memories.clear()
        if self.persist_path:
            self.save_memories()

        log_memory_operation("clear", memory_count=memory_count)
        self.logger.info("All memories cleared successfully")

    def save_memories(self) -> None:
        """Save memories to persistent storage"""
        if not self.persist_path:
            return

        try:
            self.logger.debug(
                "Saving memories to persistent storage",
                extra={
                    "persist_path": self.persist_path,
                    "memory_count": len(self.memories),
                },
            )

            # Create directory if it doesn't exist
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)

            # Convert memories to serializable format
            memories_data = [memory.to_dict() for memory in self.memories]

            # Save to JSON file
            with open(self.persist_path, "w") as f:
                json.dump(memories_data, f, indent=2)

            self.logger.info(
                "Memories saved successfully",
                extra={
                    "persist_path": self.persist_path,
                    "memory_count": len(self.memories),
                },
            )

        except Exception as e:
            self.logger.error(
                "Failed to save memories",
                extra={"persist_path": self.persist_path, "error": str(e)},
                exc_info=True,
            )
            raise

    def load_memories(self) -> None:
        """Load memories from persistent storage"""
        if not self.persist_path or not os.path.exists(self.persist_path):
            self.logger.info(
                "No existing memories to load",
                extra={"persist_path": self.persist_path},
            )
            return

        try:
            self.logger.info(
                "Loading memories from persistent storage",
                extra={"persist_path": self.persist_path},
            )

            with open(self.persist_path, "r") as f:
                memories_data = json.load(f)

            # Recreate Memory objects
            self.memories = [Memory.from_dict(data) for data in memories_data]

            # Re-add to vector store if available
            if self.vector_store and self.embedding_provider:
                self.logger.debug("Re-adding memories to vector store")
                for memory in self.memories:
                    if memory.embedding is None:
                        self.logger.debug(
                            "Generating missing embedding for loaded memory"
                        )
                        memory.embedding = self.embedding_provider.embed_text(
                            memory.content
                        )
                    self.vector_store.add_memory(memory)

            log_memory_operation("load", memory_count=len(self.memories))
            self.logger.info(
                "Memories loaded successfully",
                extra={
                    "memory_count": len(self.memories),
                    "persist_path": self.persist_path,
                },
            )

        except Exception as e:
            self.logger.error(
                "Error loading memories",
                extra={"persist_path": self.persist_path, "error": str(e)},
                exc_info=True,
            )
            self.memories = []

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        if not self.memories:
            return {
                "total_memories": 0,
                "oldest_memory": None,
                "newest_memory": None,
                "memory_types": {},
                "average_content_length": 0,
            }

        sorted_memories = sorted(self.memories, key=lambda m: m.timestamp)
        memory_types = {}
        total_content_length = 0

        for memory in self.memories:
            memory_type = memory.metadata.get("type", "unknown")
            memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
            total_content_length += len(memory.content)

        return {
            "total_memories": len(self.memories),
            "oldest_memory": sorted_memories[0].timestamp,
            "newest_memory": sorted_memories[-1].timestamp,
            "memory_types": memory_types,
            "average_content_length": total_content_length / len(self.memories),
            "total_content_length": total_content_length,
        }
