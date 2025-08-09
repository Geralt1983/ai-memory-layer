from __future__ import annotations
import os
from dataclasses import dataclass

from .providers.base import EmbeddingConfig, EmbeddingProvider
from .providers import registry as preg

ProviderName = str  # now open-ended via plugins

@dataclass(frozen=True)
class AppConfig:
    provider: ProviderName = os.getenv("EMBED_PROVIDER", "voyage")
    model: str = os.getenv("EMBED_MODEL", "voyage-2")
    dim: int = int(os.getenv("EMBED_DIM", "1536"))
    normalize: bool = True
    timeout_s: float = float(os.getenv("EMBED_TIMEOUT_S", "30"))
    max_batch_size: int = int(os.getenv("EMBED_BATCH", "128"))
    max_retries: int = int(os.getenv("EMBED_RETRIES", "3"))

def build_provider(cfg: AppConfig) -> EmbeddingProvider:
    factory = preg.get(cfg.provider)
    if not factory:
        # Hot-reload entrypoints then retry once
        preg.load_entrypoints()
        factory = preg.get(cfg.provider)
    if not factory:
        raise ValueError(f"Unknown provider: {cfg.provider}. Available: {preg.list_providers()}")
    ec = EmbeddingConfig(
        model=cfg.model, dim=cfg.dim, normalize=cfg.normalize,
        timeout_s=cfg.timeout_s, max_batch_size=cfg.max_batch_size, max_retries=cfg.max_retries
    )
    return factory(ec)