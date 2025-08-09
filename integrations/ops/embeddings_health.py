"""
Ops helper to check active embedding configuration at runtime.
Integrate into your web framework (FastAPI/Flask) as an internal endpoint.
"""
from __future__ import annotations
import os
from typing import Any, Dict, Optional

from integrations.embeddings_factory import (
    get_embedder,
    get_embedder_ab,   # available when A/B mode is supported
)

def _name_of(obj: Any) -> str:
    # Prefer provider mixin's provider_name() if present; else class name
    if hasattr(obj, "provider_name"):
        try:
            return obj.provider_name()  # type: ignore[attr-defined]
        except Exception:
            pass
    return obj.__class__.__name__

def _model_of(obj: Any) -> str:
    if hasattr(obj, "model_name"):
        try:
            return obj.model_name()  # type: ignore[attr-defined]
        except Exception:
            pass
    # Fallback to attribute if exposed
    return getattr(obj, "model", "") or ""

def _stats_of(obj: Any, texts: list[str]) -> Optional[Dict[str, Any]]:
    if hasattr(obj, "stats"):
        try:
            return obj.stats(texts)  # type: ignore[attr-defined]
        except Exception:
            return None
    return None

def check_embeddings_health() -> Dict[str, Any]:
    """
    Returns a structured view:
    {
      "ok": bool,
      "mode": "single" | "ab",
      "provider_env": "openai" | "...",
      "active": {"provider": "...", "model": "...", "stats": {...}|null},
      "shadow": {"provider": "...", "model": "...", "stats": {...}|null} | null,
      "error": "..." | null
    }
    """
    provider_env = os.getenv("EMBEDDING_PROVIDER", "openai")
    ab_conf = os.getenv("EMBED_AB_WRITE", "").strip()
    # prefer AB wrapper if configured; else single-provider
    mode = "ab" if ab_conf else "single"

    # build embedder
    if mode == "ab":
        emb = get_embedder_ab()
    else:
        emb = get_embedder()

    # attempt a tiny smoke call
    payload = ["ok"]
    error: Optional[str] = None
    ok = True
    try:
        _ = emb.embed(payload)
    except Exception as ex:
        ok = False
        error = str(ex)

    # Inspect for AB internals if present
    active_meta: Dict[str, Any] = {
        "provider": _name_of(emb),
        "model": _model_of(emb),
        "stats": _stats_of(emb, payload),
    }
    shadow_meta: Optional[Dict[str, Any]] = None
    # DualWriteEmbeddings exposes .primary/.shadow
    primary = getattr(emb, "primary", None)
    shadow = getattr(emb, "shadow", None)
    if primary is not None and shadow is not None:
        active_meta = {
            "provider": _name_of(primary),
            "model": _model_of(primary),
            "stats": _stats_of(primary, payload),
        }
        shadow_meta = {
            "provider": _name_of(shadow),
            "model": _model_of(shadow),
            "stats": _stats_of(shadow, payload),
        }

    return {
        "ok": ok,
        "mode": mode,
        "provider_env": provider_env,
        "active": active_meta,
        "shadow": shadow_meta,
        "error": error,
    }