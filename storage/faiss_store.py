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
    class MockFaiss:
        METRIC_L2 = 1

        @staticmethod
        def IndexFlatL2(dim):
            return MockIndex()

    class MockIndex:
        def __init__(self):
            self.ntotal = 0
            self.d = 0

        def add(self, vectors):
            pass

        def search(self, query, k):
            return [[]], [[]]

        def reset(self):
            pass

    faiss = MockFaiss()
from typing import List, Dict, Any, Optional
import pickle
import os
from core.memory_engine import Memory, VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, dimension: int = 1536, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.memories: Dict[int, Memory] = {}
        self.current_id = 0
        self.index_path = index_path

        if index_path and os.path.exists(f"{index_path}.index"):
            self.load_index(index_path)

    def add_memory(self, memory: Memory) -> str:
        if memory.embedding is None:
            raise ValueError(
                "Memory must have an embedding to be added to vector store"
            )

        embedding = memory.embedding.astype("float32").reshape(1, -1)
        self.index.add(embedding)

        memory_id = str(self.current_id)
        self.memories[self.current_id] = memory
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
        for i, idx in enumerate(indices[0]):
            if idx in self.memories:
                memory = self.memories[idx]
                memory.relevance_score = float(1 / (1 + distances[0][i]))
                results.append(memory)

        return results

    def delete_memory(self, memory_id: str) -> bool:
        try:
            idx = int(memory_id)
            if idx in self.memories:
                del self.memories[idx]
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
