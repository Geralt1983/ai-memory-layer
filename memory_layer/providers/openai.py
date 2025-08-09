from __future__ import annotations
import os, time
from typing import Sequence, List
import requests

from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingError, ProviderUnavailable

_OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
_EMBED_URL = f"{_OPENAI_BASE}/embeddings"

class OpenAIEmbeddings(EmbeddingProvider):
    """
    Minimal OpenAI embeddings provider compatible with EmbeddingProvider.
    Env:
      OPENAI_API_KEY  (required)
      OPENAI_BASE_URL (optional; defaults to official)
      OPENAI_ORG_ID   (optional)
    """
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__(cfg)
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.org = os.getenv("OPENAI_ORG_ID", "")
        self.session = requests.Session()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.org:
            headers["OpenAI-Organization"] = self.org
        self.session.headers.update(headers)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.is_available():
            raise ProviderUnavailable("OPENAI_API_KEY not set")
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
                    resp = self.session.post(_EMBED_URL, json=payload, timeout=self.cfg.timeout_s)
                    # Basic retry on 5xx or 429
                    if resp.status_code in (429, 500, 502, 503, 504) and tries < self.cfg.max_retries:
                        time.sleep(min(1.0 * tries, 5.0))
                        continue
                    resp.raise_for_status()
                    vecs = [r["embedding"] for r in resp.json()["data"]]
                    out.extend(vecs)
                    break
                except requests.Timeout as e:
                    if tries >= self.cfg.max_retries:
                        raise EmbeddingError(f"openai timeout after {tries} tries") from e
                    time.sleep(0.5 * tries)
                except requests.RequestException as e:
                    raise EmbeddingError(f"openai error: {e}") from e
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