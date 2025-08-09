from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Type, Optional
import importlib
import importlib.metadata as im

from .base import EmbeddingProvider, EmbeddingConfig

_REGISTRY: Dict[str, Callable[[EmbeddingConfig], EmbeddingProvider]] = {}

def register(name: str):
    """Decorator for runtime registration."""
    def _wrap(cls: Type[EmbeddingProvider]):
        _REGISTRY[name] = lambda cfg: cls(cfg)
        return cls
    return _wrap

def load_entrypoints(group: str = "ai_memory_layer.providers") -> None:
    """Load providers from Python entry points."""
    try:
        for ep in im.entry_points().select(group=group):
            factory = ep.load()
            if not callable(factory):  # factory(cfg) -> provider
                continue
            _REGISTRY[ep.name] = factory
    except Exception:
        pass

def get(name: str) -> Optional[Callable[[EmbeddingConfig], EmbeddingProvider]]:
    return _REGISTRY.get(name)

def list_providers() -> list[str]:
    return sorted(_REGISTRY.keys())

def register_builtin_providers() -> None:
    """Built-ins without creating import cycles."""
    # Lazy import to avoid importing openai/voyage if not used
    from .voyage import VoyageEmbeddings
    try:
        from .openai import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None  # optional

    _REGISTRY.setdefault("voyage", lambda cfg: VoyageEmbeddings(cfg))
    if OpenAIEmbeddings:
        _REGISTRY.setdefault("openai", lambda cfg: OpenAIEmbeddings(cfg))

# Initialize once
register_builtin_providers()
load_entrypoints()