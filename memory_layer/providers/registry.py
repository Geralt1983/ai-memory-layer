from __future__ import annotations
from typing import Callable, Dict, Type, Optional
import importlib.metadata as im
import logging

from .base import EmbeddingProvider, EmbeddingConfig

log = logging.getLogger(__name__)
_REGISTRY: Dict[str, Callable[[EmbeddingConfig], EmbeddingProvider]] = {}

def register(name: str):
    """Decorator for runtime registration."""
    def _wrap(cls: Type[EmbeddingProvider]):
        if name in _REGISTRY:
            raise ValueError(f"Provider '{name}' already registered")
        _REGISTRY[name] = lambda cfg: cls(cfg)
        log.debug("Registered provider '%s' from %s", name, cls.__module__)
        return cls
    return _wrap

def load_entrypoints(group: str = "ai_memory_layer.providers") -> None:
    try:
        for ep in im.entry_points().select(group=group):
            try:
                factory = ep.load()
            except Exception as e:
                log.debug("Failed loading provider entrypoint %s: %s", ep.name, e)
                continue
            if not callable(factory):
                log.debug("Skip provider '%s': factory not callable", ep.name)
                continue
            if ep.name in _REGISTRY:
                log.debug("Skip provider '%s': already registered", ep.name)
                continue
            _REGISTRY[ep.name] = factory
            log.debug("Loaded provider '%s' from entrypoint %s", ep.name, ep.value)
    except Exception as e:
        log.debug("Entry point scan failed: %s", e)

def get(name: str) -> Optional[Callable[[EmbeddingConfig], EmbeddingProvider]]:
    return _REGISTRY.get(name)

def list_providers() -> list[str]:
    return sorted(_REGISTRY.keys())

def register_builtin_providers() -> None:
    from .voyage import VoyageEmbeddings
    try:
        from .openai import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None
    _REGISTRY.setdefault("voyage", lambda cfg: VoyageEmbeddings(cfg))
    if OpenAIEmbeddings:
        _REGISTRY.setdefault("openai", lambda cfg: OpenAIEmbeddings(cfg))

# Initialize once
register_builtin_providers()
load_entrypoints()