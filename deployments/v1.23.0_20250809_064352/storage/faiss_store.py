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
        
        @staticmethod
        def asarray(data, dtype=None):
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
        
        @staticmethod
        def IndexFlatIP(dim):
            return MockIndex()
        
        @staticmethod
        def read_index(path):
            return MockIndex()
        
        @staticmethod
        def write_index(index, path):
            pass

    class MockIndex:
        def __init__(self):
            self.ntotal = 0
            self.d = 1536

        def add(self, vectors):
            pass

        def search(self, query, k):
            return [[]], [[]]

        def reset(self):
            pass

    faiss = MockFaiss()

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import os
from core.memory_engine import Memory, VectorStore

@dataclass
class IndexMeta:
    dim: int
    next_id: int


class FaissVectorStore(VectorStore):
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.meta_path = f"{index_path}.meta.json"
        self.idmap_path = f"{index_path}.ids.json"
        self.current_id = 0
        self.id_to_mem = {}  # id -> memory_id
        self.memories: Dict[int, Memory] = {}

        if os.path.exists(f"{index_path}.index"):
            self.load_index(index_path)
        else:
            self.index = faiss.IndexFlatIP(1536)  # default, updated on first add
            self._persist_meta(IndexMeta(dim=1536, next_id=0))
            self._persist_idmap()

    def _persist_meta(self, meta: IndexMeta):
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        with open(self.meta_path, "w") as f:
            json.dump(asdict(meta), f)

    def _load_meta(self) -> IndexMeta:
        with open(self.meta_path) as f:
            d = json.load(f)
        return IndexMeta(**d)

    def _persist_idmap(self):
        os.makedirs(os.path.dirname(self.idmap_path), exist_ok=True)
        with open(self.idmap_path, "w") as f:
            json.dump(self.id_to_mem, f)

    def _load_idmap(self):
        if os.path.exists(self.idmap_path):
            with open(self.idmap_path) as f:
                self.id_to_mem = json.load(f)

    def load_index(self, index_path: str):
        self.index = faiss.read_index(f"{index_path}.index")
        meta = self._load_meta()
        self.current_id = meta.next_id
        self._load_idmap()
        
        # Load memories from legacy .pkl if exists, otherwise from .ids.json
        pkl_path = f"{index_path}.pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.memories = data.get("memories", {})

    def save_index(self):
        faiss.write_index(self.index, f"{self.index_path}.index")
        self._persist_meta(IndexMeta(dim=self.index.d, next_id=self.current_id))
        self._persist_idmap()
        
        # Keep legacy .pkl for backward compatibility
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump(
                {
                    "memories": self.memories,
                    "current_id": self.current_id,
                    "dimension": self.index.d,
                },
                f,
            )

    def _ensure_dim(self, vec):
        if not FAISS_AVAILABLE or not NUMPY_AVAILABLE:
            return
        vec = np.asarray(vec, dtype="float32")
        if len(vec.shape) == 1:
            vec = vec.reshape(1, -1)
        if self.index.d != vec.shape[1]:
            # rebuild index if mismatch
            if self.index.ntotal > 0:
                raise ValueError(f"Embedding dimension changed from {self.index.d} to {vec.shape[1]}; cannot migrate a non-empty index automatically.")
            self.index = faiss.IndexFlatIP(vec.shape[1])

    def add_memory(self, memory: Memory) -> str:
        if memory.embedding is None:
            raise ValueError(
                "Memory must have an embedding to be added to vector store"
            )

        if NUMPY_AVAILABLE:
            emb = np.asarray(memory.embedding, dtype="float32").reshape(1, -1)
            self._ensure_dim(emb)
            self.index.add(emb)
        
        memory_id = str(self.current_id)
        self.memories[self.current_id] = memory
        self.id_to_mem[memory_id] = memory.id
        self.current_id += 1

        self.save_index()
        return memory_id

    def search(self, query_embedding, k: int = 5) -> List[Memory]:
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        
        if NUMPY_AVAILABLE:
            query_embedding = np.asarray(query_embedding, dtype="float32").reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx in self.memories:
                memory = self.memories[idx]
                # Use dot product for IP index (higher is better)
                memory.relevance_score = float(distances[0][i])
                results.append(memory)

        return results

    def delete_memory(self, memory_id: str) -> bool:
        try:
            idx = int(memory_id)
            if idx in self.memories:
                del self.memories[idx]
                if memory_id in self.id_to_mem:
                    del self.id_to_mem[memory_id]
                self.save_index()
                return True
        except ValueError:
            pass
        return False
