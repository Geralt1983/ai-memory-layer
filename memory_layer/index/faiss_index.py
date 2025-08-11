from __future__ import annotations
import os, json, hashlib
from dataclasses import dataclass
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
    metric: str = "IP"  # use inner product for normalized vectors

def _meta_digest(spec: IndexSpec, corpus_hashes: List[str]) -> str:
    payload = {"spec": spec.__dict__, "corpus": corpus_hashes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

class FAISSIndex:
    def __init__(self, path: str, spec: IndexSpec):
        self.path = path
        self.spec = spec

    def _metric(self):
        return faiss.METRIC_INNER_PRODUCT if self.spec.metric == "IP" else faiss.METRIC_L2

    def load_or_build(self, vectors: np.ndarray, ids: List[str]) -> faiss.Index:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        meta_path = os.path.join(os.path.dirname(self.path), META_FILE)
        digest = _meta_digest(self.spec, ids)
        # Fast path: load if index + meta exist and digest matches
        if os.path.exists(self.path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                if meta.get("digest") == digest:
                    return faiss.read_index(self.path)
            except Exception:
                pass
        # Build
        index = faiss.IndexFlatIP(self.spec.dim) if self._metric()==faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.spec.dim)
        index.add(vectors.astype("float32"))
        faiss.write_index(index, self.path)
        with open(meta_path, "w") as f:
            json.dump({"digest": digest, "count": int(index.ntotal), "spec": self.spec.__dict__}, f)
        return index

    @staticmethod
    def search(index: faiss.Index, queries: np.ndarray, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        return index.search(queries.astype("float32"), k)