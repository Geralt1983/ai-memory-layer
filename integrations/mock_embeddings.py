from __future__ import annotations
"""Mock embedding provider for tests.

This module implements a lightweight :class:`MockEmbeddings` class that
produces deterministic numerical vectors for a given input text.  It is used
by the test-suite to avoid external OpenAI dependencies and to ensure that
memory related features can be exercised in an isolated environment.
"""

from typing import List, Union
import numpy as np
import hashlib

from .embeddings import EmbeddingProvider


class MockEmbeddings(EmbeddingProvider):
    """Simple deterministic embedding generator.

    The implementation hashes the input text and converts the digest into a
    fixed-size ``numpy`` array.  It is **not** intended to produce meaningful
    semantic embeddings – only to provide stable shapes for tests.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _hash(self, text: str) -> np.ndarray:
        # Generate enough bytes by repeatedly hashing the digest
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        data = bytearray()
        while len(data) < self.dimension:
            data.extend(digest)
            digest = hashlib.sha256(digest).digest()
        # Convert bytes to float32 vector in range [0, 1]
        arr = np.frombuffer(bytes(data[: self.dimension]), dtype=np.uint8).astype(
            np.float32
        ) / 255.0
        return arr

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:  # type: ignore[override]
        if isinstance(text, list):
            text = " ".join(text)
        return self._hash(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:  # type: ignore[override]
        return [self.embed_text(t) for t in texts]
