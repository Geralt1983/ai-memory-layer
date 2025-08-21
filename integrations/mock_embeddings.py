"""Mock embedding provider used in tests.

The real project relies on external embedding services.  For unit tests we
only need deterministic vectors with the correct interface, so this module
provides a very small :class:`MockEmbeddings` implementation.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from .embeddings import EmbeddingProvider


class MockEmbeddings(EmbeddingProvider):
    """Return simple deterministic embeddings for testing."""

    def _embed(self, text: str) -> np.ndarray:
        # Produce a small deterministic vector based on Python's hash.  The
        # dimension is intentionally tiny as similarity isn't important for
        # the unit tests.
        h = hash(text) % 1000
        return np.array([float(h)] * 10, dtype="float32")

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, list):
            text = " ".join(text)
        return self._embed(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self._embed(t) for t in texts]

