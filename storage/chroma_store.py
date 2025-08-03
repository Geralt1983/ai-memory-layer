# Mock implementations for compatibility
CHROMADB_AVAILABLE = False
NUMPY_AVAILABLE = False

class MockChromaClient:
    def get_or_create_collection(self, name): return MockCollection()
    def delete_collection(self, name): pass

class MockCollection:
    def add(self, ids, documents, metadatas=None, embeddings=None): pass
    def query(self, query_embeddings=None, query_texts=None, n_results=10): 
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    def delete(self): pass
    def count(self): return 0

class MockSettings:
    def __init__(self, **kwargs): pass

class MockArray:
    def __init__(self, data=None):
        self.data = data if data is not None else []
    def tolist(self):
        return self.data if isinstance(self.data, list) else [self.data]

class MockNumpy:
    ndarray = MockArray
    @staticmethod
    def array(data): return MockArray(data)

chromadb = type('chromadb', (), {'Client': MockChromaClient, 'config': type('config', (), {'Settings': MockSettings})})
np = MockNumpy()

from typing import List, Dict, Any, Optional
import uuid
from core.memory_engine import Memory, VectorStore


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