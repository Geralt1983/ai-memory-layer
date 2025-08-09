from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Protocol

class EmbeddingError(RuntimeError): ...
class ProviderUnavailable(RuntimeError): ...

@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    dim: int
    normalize: bool = True
    timeout_s: float = 30.0
    max_batch_size: int = 128
    max_retries: int = 3

class EmbeddingProvider(ABC):
    """Stable contract for all providers."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap local check (env/config present). Can optionally do a 0-cost remote ping."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Return len(texts) vectors. Must respect cfg.max_batch_size and cfg.timeout_s."""

    def embed_query(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]