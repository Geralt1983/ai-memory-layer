from __future__ import annotations
import os, json, hashlib
from dataclasses import dataclass, asdict
from typing import List, Tuple
import faiss
import numpy as np

META_FILE = "faiss.index.meta.json"

@dataclass(frozen=True)
class IndexSpec:
    provider: str
    model: str
    dim: int
    normalize: bool
    metric: str = "IP"  # "IP" or "L2"

class FAISSIndex:
    def __init__(self, path: str, spec: IndexSpec):
        self.path = path
        self.spec = spec

    def _metric(self):
        return faiss.METRIC_INNER_PRODUCT if self.spec.metric.upper() == "IP" else faiss.METRIC_L2

    def _digest(self, vectors: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(vectors.astype("float32").tobytes())
        h.update(json.dumps(asdict(self.spec), sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def load_or_build(self, vectors: np.ndarray, ids: List[str]) -> faiss.Index:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        meta_path = os.path.join(os.path.dirname(self.path), META_FILE)
        digest = self._digest(vectors)
        if os.path.exists(self.path) and os.path.exists(meta_path):
            try:
                meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
                if meta.get("digest") == digest:
                    return faiss.read_index(self.path)
            except Exception:
                pass
        # Build flat index
        if self._metric() == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(self.spec.dim)
        else:
            index = faiss.IndexFlatL2(self.spec.dim)
        index.add(vectors.astype("float32"))
        faiss.write_index(index, self.path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"digest": digest, "count": int(index.ntotal), "spec": asdict(self.spec)}, f)
        return index

    @staticmethod
    def search(index: faiss.Index, queries: np.ndarray, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        return index.search(queries.astype("float32"), k)