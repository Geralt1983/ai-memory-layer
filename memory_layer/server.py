from __future__ import annotations
import os, json, time, pathlib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import faiss

from .config import AppConfig, build_provider
from .cache.embedding_cache import EmbeddingCache
from .index.faiss_index import FAISSIndex, IndexSpec
from .retrieval.scoring import blend

app = FastAPI(title="AI Memory Layer Search")

INDEX_DIR = os.getenv("MEM_INDEX_DIR", ".index")
CACHE_PATH = os.getenv("EMBED_CACHE_PATH", ".cache/embeddings.sqlite3")

class BuildRequest(BaseModel):
    dir: str | None = None
    jsonl: str | None = None
    out: str = INDEX_DIR
    provider: str | None = None
    model: str | None = None
    dim: int | None = None

class SearchResponse(BaseModel):
    id: str
    score: float
    text: str | None = None

@app.post("/build")
def build(req: BuildRequest):
    from .cli import _read_texts_from_dir, _read_texts_from_jsonl, _ensure_vectors
    texts = []
    if req.dir:   texts.extend(list(_read_texts_from_dir(req.dir)))
    if req.jsonl: texts.extend(list(_read_texts_from_jsonl(req.jsonl)))
    if not texts:
        raise HTTPException(400, "no input texts")

    cfg = AppConfig(
        provider=req.provider or os.getenv("EMBED_PROVIDER", "voyage"),
        model=req.model or os.getenv("EMBED_MODEL", "voyage-2"),
        dim=req.dim or int(os.getenv("EMBED_DIM", "1536")),
    )
    provider = build_provider(cfg)
    if not provider.is_available():
        raise HTTPException(400, f"provider {cfg.provider} unavailable")

    cache = EmbeddingCache(CACHE_PATH)
    ids, vecs = _ensure_vectors(texts, provider, cache, cfg.provider)
    pathlib.Path(req.out).mkdir(parents=True, exist_ok=True)
    spec = IndexSpec(provider=cfg.provider, model=cfg.model, dim=cfg.dim, normalize=cfg.normalize)
    idx = FAISSIndex(os.path.join(req.out, "faiss.index"), spec)
    idx.load_or_build(vecs, ids)
    # persist corpus + ids
    with open(os.path.join(req.out, "ids.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(ids))
    with open(os.path.join(req.out, "corpus.jsonl"), "w", encoding="utf-8") as f:
        now = time.time()
        for i, did in enumerate(ids):
            f.write(json.dumps({"id": did, "text": texts[i], "ts": now, "tags": 0, "thread_len": 1}) + "\n")
    return {"count": len(ids), "out": req.out}

@app.get("/search", response_model=list[SearchResponse])
def search(q: str = Query(..., description="query text"),
           k: int = 8,
           out: str = INDEX_DIR,
           provider: str | None = None,
           model: str | None = None,
           dim: int | None = None):
    cfg = AppConfig(
        provider=provider or os.getenv("EMBED_PROVIDER", "voyage"),
        model=model or os.getenv("EMBED_MODEL", "voyage-2"),
        dim=dim or int(os.getenv("EMBED_DIM", "1536")),
    )
    provider = build_provider(cfg)
    if not provider.is_available():
        raise HTTPException(400, f"provider {cfg.provider} unavailable")
    qv = np.array([provider.embed_query(q)], dtype="float32")
    index_path = os.path.join(out, "faiss.index")
    if not os.path.exists(index_path):
        raise HTTPException(404, f"index not found at {index_path}")
    index = faiss.read_index(index_path)
    D, I = index.search(qv, max(32, k))
    ids_path = os.path.join(out, "ids.json")
    if not os.path.exists(ids_path):
        raise HTTPException(500, "ids.json missing")
    ids_list = json.loads(open(ids_path, "r", encoding="utf-8").read())
    # load corpus
    corpus = {}
    cpath = os.path.join(out, "corpus.jsonl")
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                corpus[obj["id"]] = obj

    now = time.time()
    pairs = []
    for local_idx, sem in zip(I[0].tolist(), D[0].tolist()):
        if 0 <= local_idx < len(ids_list):
            did = ids_list[local_idx]
            meta = corpus.get(did, {})
            score = blend(
                semantic=float(sem),
                days_old=max(0.0, (now - float(meta.get("ts", now))) / 86400.0),
                tags=int(meta.get("tags", 0)),
                thread_len=int(meta.get("thread_len", 1)),
            )
            pairs.append((did, score))
    pairs.sort(key=lambda x: x[1], reverse=True)
    out_list = []
    for did, score in pairs[:k]:
        out_list.append(SearchResponse(id=did, score=score, text=corpus.get(did, {}).get("text")))
    return out_list

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/providers")
def list_providers():
    """List available embedding providers."""
    from .providers import registry as preg
    return {"providers": preg.list_providers()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8080")))