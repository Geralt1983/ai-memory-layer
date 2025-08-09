from __future__ import annotations
import os, time
from typing import Sequence, List
import requests
from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingError, ProviderUnavailable

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

class VoyageEmbeddings(EmbeddingProvider):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__(cfg)
        self.api_key = os.getenv("VOYAGE_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def is_available(self) -> bool:
        return bool(self.api_key)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.is_available():
            raise ProviderUnavailable("VOYAGE_API_KEY not set")
        if not texts:
            return []
        out: List[List[float]] = []
        bs = self.cfg.max_batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            payload = {"model": self.cfg.model, "input": list(chunk)}
            tries = 0
            while True:
                tries += 1
                try:
                    resp = self.session.post(VOYAGE_URL, json=payload, timeout=self.cfg.timeout_s)
                    # Retry on 5xx with jitter
                    if resp.status_code >= 500 and tries < self.cfg.max_retries:
                        import random
                        jitter = random.uniform(0.1, 0.5)
                        time.sleep(0.5 * tries + jitter)
                        continue
                    resp.raise_for_status()
                    vecs = [r["embedding"] for r in resp.json()["data"]]
                    out.extend(vecs)
                    break
                except requests.Timeout as e:
                    if tries >= self.cfg.max_retries:
                        raise EmbeddingError(f"voyage timeout after {tries} tries") from e
                    import random
                    jitter = random.uniform(0.1, 0.3)
                    time.sleep(0.5 * tries + jitter)
                except requests.RequestException as e:
                    raise EmbeddingError(f"voyage error: {e}") from e
        if self.cfg.normalize:
            out = _l2_norm(out)
        return out

def _l2_norm(vecs: List[List[float]]) -> List[List[float]]:
    import math
    normed = []
    for v in vecs:
        n = math.sqrt(sum(x*x for x in v)) or 1.0
        normed.append([x/n for x in v])
    return normed