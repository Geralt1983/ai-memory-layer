from __future__ import annotations
"""Simple in-memory vector store used for tests.

This module provides a minimal :class:`MockVectorStore` that implements the
subset of the :class:`~core.memory_engine.VectorStore` interface required by the
unit tests.  It allows tests to run without requiring an actual vector database
backend.
"""

from typing import List
from core.memory_engine import Memory, VectorStore


class MockVectorStore(VectorStore):
    """A trivial vector store that keeps memories in a list."""

    def __init__(self) -> None:
        self.memories: List[Memory] = []

    def add_memory(self, memory: Memory) -> str:
        self.memories.append(memory)
        # Use list index as identifier
        return str(len(self.memories) - 1)

    def search(self, query_embedding, k: int = 5) -> List[Memory]:
        # Return up to ``k`` memories in FIFO order; no similarity used
        return self.memories[:k]

    def delete_memory(self, memory_id: str) -> bool:
        try:
            idx = int(memory_id)
            self.memories.pop(idx)
            return True
        except Exception:
            return False

    def clear_all(self) -> None:
        self.memories.clear()
