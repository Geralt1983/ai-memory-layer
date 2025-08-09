from __future__ import annotations
import json
import os
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Depends, Request
from pydantic import BaseModel
import numpy as np
import faiss

from .config import AppConfig, build_provider
from .cache.embedding_cache import EmbeddingCache
from .index.faiss_index import FAISSIndex, IndexSpec
from .retrieval.scoring import blend
from .providers import registry as preg
from .providers.base import EmbeddingConfig

app = FastAPI(title="AI Memory Layer Search")

INDEX_DIR = Path(os.getenv("MEM_INDEX_DIR", ".index")).resolve()
CACHE_PATH = os.getenv("EMBED_CACHE_PATH", ".cache/embeddings.sqlite3")
SERVER_API_KEY = os.getenv("SERVER_API_KEY", "")


def _require_api_key(req: Request):
    if not SERVER_API_KEY:
        return
    if req.headers.get("x-api-key") != SERVER_API_KEY:
        raise HTTPException(403, "Invalid API key")


def _safe_out(path_str: str) -> Path:
    target = (
        (INDEX_DIR / path_str).resolve()
        if not Path(path_str).is_absolute()
        else Path(path_str).resolve()
    )
    if not str(target).startswith(str(INDEX_DIR)):
        raise HTTPException(400, "out path must be under MEM_INDEX_DIR")
    target.mkdir(parents=True, exist_ok=True)
    return target


class BuildRequest(BaseModel):
    dir: str | None = None
    jsonl: str | None = None
    out: str = str(INDEX_DIR)
    provider: str | None = None
    model: str | None = None
    dim: int | None = None


class SearchResponse(BaseModel):
    id: str
    score: float
    text: str | None = None


class ProviderInfo(BaseModel):
    name: str
    available: bool
    error: str | None = None
    models: list[str] | None = None


class ProviderHealthResponse(BaseModel):
    name: str
    available: bool
    response_time_ms: float | None = None
    error: str | None = None


class AutoSelectResponse(BaseModel):
    selected_provider: str
    reason: str
    all_providers: list[ProviderInfo]


@app.post("/build")
def build(req: BuildRequest, _: None = Depends(_require_api_key)):
    from .cli import _read_texts_from_dir, _read_texts_from_jsonl, _ensure_vectors

    texts: list[str] = []
    if req.dir:
        texts.extend(list(_read_texts_from_dir(req.dir)))
    if req.jsonl:
        texts.extend(list(_read_texts_from_jsonl(req.jsonl)))
    if not texts:
        raise HTTPException(400, "no input texts")

    # Auto-select provider if not specified
    provider_name = req.provider
    if not provider_name:
        provider_name = os.getenv("EMBED_PROVIDER")
        if not provider_name:
            # Use autodiscovery to select best available provider
            auto_result = auto_select_provider()
            provider_name = auto_result.selected_provider
            if provider_name == "none":
                msg = "No embedding providers available"
                raise HTTPException(400, msg)

    cfg = AppConfig(
        provider=provider_name,
        model=req.model or os.getenv("EMBED_MODEL", "voyage-2"),
        dim=req.dim or int(os.getenv("EMBED_DIM", "1536")),
    )
    provider = build_provider(cfg)
    if not provider.is_available():
        raise HTTPException(400, f"provider {cfg.provider} unavailable")

    cache = EmbeddingCache(CACHE_PATH)
    ids, vecs = _ensure_vectors(texts, provider, cache, cfg.provider)
    out_dir = _safe_out(req.out)
    spec = IndexSpec(
        provider=cfg.provider,
        model=cfg.model,
        dim=cfg.dim,
        normalize=cfg.normalize,
        metric="IP" if cfg.normalize else "L2",
    )
    idx = FAISSIndex(str(out_dir / "faiss.index"), spec)
    idx.load_or_build(vecs, ids)

    # persist corpus + ids
    (out_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    now = time.time()
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for i, did in enumerate(ids):
            f.write(
                json.dumps(
                    {"id": did, "text": texts[i], "ts": now, "tags": 0, "thread_len": 1}
                )
                + "\n"
            )
    return {"count": len(ids), "out": str(out_dir)}


@app.get("/search", response_model=list[SearchResponse])
def search(
    q: str = Query(..., description="query text"),
    k: int = 8,
    out: str = str(INDEX_DIR),
    provider: str | None = None,
    model: str | None = None,
    dim: int | None = None,
):
    # Auto-select provider if not specified
    provider_name = provider
    if not provider_name:
        provider_name = os.getenv("EMBED_PROVIDER")
        if not provider_name:
            # Use autodiscovery to select best available provider
            auto_result = auto_select_provider()
            provider_name = auto_result.selected_provider
            if provider_name == "none":
                msg = "No embedding providers available"
                raise HTTPException(400, msg)

    cfg = AppConfig(
        provider=provider_name,
        model=model or os.getenv("EMBED_MODEL", "voyage-2"),
        dim=dim or int(os.getenv("EMBED_DIM", "1536")),
    )
    prov = build_provider(cfg)
    if not prov.is_available():
        raise HTTPException(400, f"provider {cfg.provider} unavailable")
    qv = np.array([prov.embed_query(q)], dtype="float32")

    out_dir = _safe_out(out)
    index_path = out_dir / "faiss.index"
    ids_path = out_dir / "ids.json"
    corpus_path = out_dir / "corpus.jsonl"

    if not index_path.exists():
        raise HTTPException(404, f"index not found at {index_path}")
    index = faiss.read_index(str(index_path))
    distances, indices = index.search(qv, max(32, k))

    ids_list = (
        json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.exists() else []
    )
    meta = {}
    if corpus_path.exists():
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                meta[obj["id"]] = obj

    pairs = []
    for rank, idx_i in enumerate(indices[0]):
        did = ids_list[idx_i] if idx_i < len(ids_list) else str(idx_i)
        raw = float(distances[0][rank])
        rec = meta.get(did, {})
        ts = float(rec.get("ts", time.time()))
        days_old = max(0.0, (time.time() - ts) / 86400.0)
        score = blend(
            semantic=raw,
            days_old=days_old,
            tags=int(rec.get("tags", 0)),
            thread_len=int(rec.get("thread_len", 1)),
        )
        pairs.append((did, score))

    pairs.sort(key=lambda x: x[1], reverse=True)
    out_list = [
        SearchResponse(id=did, score=sc, text=meta.get(did, {}).get("text"))
        for did, sc in pairs[:k]
    ]
    return out_list


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/providers")
def list_providers():
    return {"providers": preg.list_providers()}


@app.get("/providers/discover", response_model=list[ProviderInfo])
def discover_providers():
    """Auto-discover all available embedding providers."""
    # Force reload of entry points to discover new providers
    preg.load_entrypoints()

    providers = []
    for name in preg.list_providers():
        provider_info = ProviderInfo(name=name, available=False)

        try:
            factory = preg.get(name)
            if not factory:
                provider_info.error = "Factory not found"
                providers.append(provider_info)
                continue

            # Try to create provider with default config
            default_config = EmbeddingConfig(
                model="default",
                dim=1536,
                normalize=True,
                timeout_s=30.0,
                max_batch_size=128,
                max_retries=3,
            )

            provider = factory(default_config)
            provider_info.available = provider.is_available()

            if not provider_info.available:
                provider_info.error = "Provider reported unavailable"
            else:
                # Try to discover supported models if possible
                if hasattr(provider, "supported_models"):
                    provider_info.models = provider.supported_models()

        except Exception as e:
            provider_info.error = str(e)

        providers.append(provider_info)

    return providers


@app.get("/providers/{provider_name}/health", response_model=ProviderHealthResponse)
def check_provider_health(provider_name: str, model: str = "default"):
    """Check the health and availability of a specific provider."""
    import time

    response = ProviderHealthResponse(name=provider_name, available=False)

    try:
        factory = preg.get(provider_name)
        if not factory:
            response.error = "Provider not found"
            return response

        config = EmbeddingConfig(
            model=model,
            dim=1536,
            normalize=True,
            timeout_s=30.0,
            max_batch_size=128,
            max_retries=3,
        )

        provider = factory(config)

        start_time = time.time()
        response.available = provider.is_available()
        end_time = time.time()

        response.response_time_ms = (end_time - start_time) * 1000

        if not response.available:
            response.error = "Provider reported unavailable"

    except Exception as e:
        response.error = str(e)

    return response


@app.get("/providers/auto-select", response_model=AutoSelectResponse)
def auto_select_provider():
    """Automatically select the best available provider."""
    # Force reload of entry points to discover new providers
    preg.load_entrypoints()

    # Provider preference order (can be configured via environment)
    provider_priority = os.getenv("PROVIDER_PRIORITY", "openai,voyage").split(",")
    provider_priority = [p.strip() for p in provider_priority if p.strip()]

    providers = []
    available_providers = []

    for name in preg.list_providers():
        provider_info = ProviderInfo(name=name, available=False)

        try:
            factory = preg.get(name)
            if factory:
                default_config = EmbeddingConfig(
                    model="default",
                    dim=1536,
                    normalize=True,
                    timeout_s=30.0,
                    max_batch_size=128,
                    max_retries=3,
                )
                provider = factory(default_config)
                provider_info.available = provider.is_available()

                if provider_info.available:
                    available_providers.append(name)
                    if hasattr(provider, "supported_models"):
                        provider_info.models = provider.supported_models()
                else:
                    provider_info.error = "Provider reported unavailable"
            else:
                provider_info.error = "Factory not found"

        except Exception as e:
            provider_info.error = str(e)

        providers.append(provider_info)

    # Select best provider based on priority and availability
    selected_provider = None
    reason = "No providers available"

    if available_providers:
        # First, try providers in priority order
        for preferred in provider_priority:
            if preferred in available_providers:
                selected_provider = preferred
                reason = f"Selected based on priority order: {provider_priority}"
                break

        # If no preferred provider is available, pick the first available one
        if not selected_provider:
            selected_provider = available_providers[0]
            reason = "Selected first available provider (no priority match)"

    if not selected_provider:
        # Fallback to first registered provider even if unavailable
        all_providers = preg.list_providers()
        if all_providers:
            selected_provider = all_providers[0]
            reason = "Fallback selection - provider may not be available"
        else:
            selected_provider = "none"
            reason = "No providers registered"

    return AutoSelectResponse(
        selected_provider=selected_provider, reason=reason, all_providers=providers
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8080")))
