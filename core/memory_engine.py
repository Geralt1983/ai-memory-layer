from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Memory:
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'relevance_score': self.relevance_score
        }


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
    def __init__(self, vector_store: Optional[VectorStore] = None, embedding_provider: Optional['EmbeddingProvider'] = None):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.memories: List[Memory] = []
        
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Memory:
        memory = Memory(
            content=content,
            metadata=metadata or {}
        )
        
        # Generate embedding if embedding provider is available
        if self.embedding_provider:
            memory.embedding = self.embedding_provider.embed_text(content)
        
        self.memories.append(memory)
        
        if self.vector_store and memory.embedding is not None:
            self.vector_store.add_memory(memory)
            
        return memory
    
    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        if not self.vector_store or not self.embedding_provider:
            # Fallback to recent memories if no vector store or embedding provider
            return self.get_recent_memories(k)
        
        # Generate embedding for the query
        query_embedding = self.embedding_provider.embed_text(query)
        
        # Search using the vector store
        results = self.vector_store.search(query_embedding, k)
        
        return results
    
    def get_recent_memories(self, n: int = 10) -> List[Memory]:
        return sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:n]
    
    def clear_memories(self) -> None:
        self.memories.clear()