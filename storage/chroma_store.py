import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
from ..core.memory_engine import Memory, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str = "memories", persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.id_to_memory: Dict[str, Memory] = {}
    
    def add_memory(self, memory: Memory) -> str:
        if memory.embedding is None:
            raise ValueError("Memory must have an embedding to be added to vector store")
        
        memory_id = str(uuid.uuid4())
        
        self.collection.add(
            embeddings=[memory.embedding.tolist()],
            documents=[memory.content],
            metadatas=[memory.metadata],
            ids=[memory_id]
        )
        
        self.id_to_memory[memory_id] = memory
        return memory_id
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Memory]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        memories = []
        if results['ids'] and results['ids'][0]:
            for i, memory_id in enumerate(results['ids'][0]):
                if memory_id in self.id_to_memory:
                    memory = self.id_to_memory[memory_id]
                else:
                    memory = Memory(
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    )
                
                if results['distances']:
                    memory.relevance_score = float(1 / (1 + results['distances'][0][i]))
                
                memories.append(memory)
        
        return memories
    
    def delete_memory(self, memory_id: str) -> bool:
        try:
            self.collection.delete(ids=[memory_id])
            if memory_id in self.id_to_memory:
                del self.id_to_memory[memory_id]
            return True
        except Exception:
            return False
    
    def clear_all(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        self.id_to_memory.clear()