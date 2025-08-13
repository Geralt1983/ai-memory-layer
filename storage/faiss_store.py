from typing import List, Dict, Any, Optional
import pickle
import os
from core.memory_engine import Memory, VectorStore

# Try to import numpy and faiss, fall back to mock if not available
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

    class MockArray:
        def __init__(self, data=None):
            self.data = data if data is not None else []

        def tolist(self):
            return self.data if isinstance(self.data, list) else [self.data]

    class MockNumpy:
        ndarray = MockArray

        @staticmethod
        def array(data):
            return MockArray(data)

    np = MockNumpy()

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

    # Mock faiss for compatibility
    class MockIndex:
        """Simple in-memory mock for FAISS indices"""

        def __init__(self, dim: int = 0):
            self.d = dim
            self.vectors: Dict[int, Any] = {}

        @property
        def ntotal(self) -> int:
            return len(self.vectors)

        def add_with_ids(self, vectors, ids):
            for vec, idx in zip(vectors, ids):
                self.vectors[int(idx)] = vec

        def search(self, query, k):
            ids = list(self.vectors.keys())[:k]
            distances = [[0.0] * len(ids)]
            return distances, [ids]

        def remove_ids(self, ids):
            removed = 0
            for idx in ids:
                if int(idx) in self.vectors:
                    del self.vectors[int(idx)]
                    removed += 1
            return removed

        def reset(self):
            self.vectors = {}

    class MockFaiss:
        METRIC_L2 = 1

        @staticmethod
        def IndexFlatL2(dim):
            return MockIndex(dim)

        @staticmethod
        def IndexIDMap(base_index):
            return base_index

        @staticmethod
        def write_index(index, path):
            with open(path, "wb") as f:
                pickle.dump(index, f)

        @staticmethod
        def read_index(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    faiss = MockFaiss()


class FaissVectorStore(VectorStore):
    def __init__(self, dimension: int = 1536, index_path: Optional[str] = None):
        self.dimension = dimension
        # Use simple FlatL2 index instead of IndexIDMap to avoid add_with_ids issues
        self.index = faiss.IndexFlatL2(dimension)
        self.memories: Dict[int, Memory] = {}
        self.current_id = 0
        self.index_path = index_path
        self.id_to_index_map: Dict[int, int] = {}  # Map memory ID to index position

        if index_path and os.path.exists(f"{index_path}.index"):
            self.load_index(index_path)

    def add_memory(self, memory: Memory) -> str:
        if memory.embedding is None:
            raise ValueError(
                "Memory must have an embedding to be added to vector store"
            )

        embedding = memory.embedding.astype("float32").reshape(1, -1)

        # Get the current index position before adding
        index_position = self.index.ntotal
        
        # Add to FAISS index
        self.index.add(embedding)

        # Map memory ID to index position
        memory_id_int = self.current_id
        self.id_to_index_map[memory_id_int] = index_position
        
        memory_id = str(memory_id_int)
        self.memories[memory_id_int] = memory
        self.current_id += 1

        if self.index_path:
            self.save_index(self.index_path)

        return memory_id

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Memory]:
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        query_embedding = query_embedding.astype("float32").reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        # Map index positions back to memory IDs
        index_to_id_map = {v: k for k, v in self.id_to_index_map.items()}
        
        for i, index_pos in enumerate(indices[0]):
            if index_pos in index_to_id_map:
                memory_id = index_to_id_map[index_pos]
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    memory.relevance_score = float(1 / (1 + distances[0][i]))
                    results.append(memory)

        return results

    def delete_memory(self, memory_id: str) -> bool:
        try:
            idx = int(memory_id)
            if idx in self.memories:
                del self.memories[idx]

                if hasattr(self.index, "remove_ids"):
                    ids = np.array([idx], dtype="int64")
                    self.index.remove_ids(ids)

                if self.index_path:
                    self.save_index(self.index_path)

                return True
        except ValueError:
            pass
        return False


    def save_index(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(
                {
                    "memories": self.memories,
                    "current_id": self.current_id,
                    "dimension": self.dimension,
                    "id_to_index_map": self.id_to_index_map,
                },
                f,
            )

    def load_index(self, path: str) -> None:
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")

        if os.path.exists(f"{path}.pkl"):
            with open(f"{path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.memories = data["memories"]
                self.current_id = data["current_id"]
                self.dimension = data["dimension"]
                # Load mapping if available, otherwise rebuild it
                self.id_to_index_map = data.get("id_to_index_map", {})
                
                # If mapping is missing, rebuild it based on memory IDs
                if not self.id_to_index_map and self.memories:
                    for i, memory_id in enumerate(sorted(self.memories.keys())):
                        self.id_to_index_map[memory_id] = i
