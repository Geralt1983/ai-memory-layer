from __future__ import annotations
import argparse, json, os, sys, pathlib, time
import numpy as np

from .config import AppConfig, build_provider
from .cache.embedding_cache import EmbeddingCache
from .index.faiss_index import FAISSIndex, IndexSpec
from .retrieval.scoring import blend

def _read_texts_from_dir(path: str):
    p = pathlib.Path(path)
    for fp in p.rglob("*"):
        if fp.suffix.lower() in {".txt", ".md", ".json"} and fp.is_file():
            try:
                yield fp.read_text(encoding="utf-8")
            except Exception:
                pass

def _read_texts_from_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            # accept {"text": "..."} or raw string line
            yield obj["text"] if isinstance(obj, dict) and "text" in obj else str(obj)

def _ensure_vectors(texts, provider, cache: EmbeddingCache, provider_name: str):
    ids = [cache.content_hash(t) for t in texts]
    hit = cache.get_many(ids)
    to_idx = [i for i, h in enumerate(ids) if h not in hit]
    new_vecs = []
    if to_idx:
        new_vecs = provider.embed_batch([texts[i] for i in to_idx])
        cache.put_many(
            (ids[i], v, provider_name, provider.cfg.model, provider.cfg.dim, int(provider.cfg.normalize))
            for i, v in zip(to_idx, new_vecs)
        )
    def get_vec(i):
        return hit.get(ids[i]) if ids[i] in hit else new_vecs[to_idx.index(i)]
    vectors = [get_vec(i) for i in range(len(texts))]
    return ids, np.array(vectors, dtype="float32")

def _write_corpus(out_dir: str, ids, texts, ts=None, tags=None, thread_len=None):
    """Write corpus metadata to JSONL for re-ranking."""
    p = pathlib.Path(out_dir) / "corpus.jsonl"
    now = time.time()
    ts = ts or [now] * len(ids)
    tags = tags or [0] * len(ids)
    thread_len = thread_len or [1] * len(ids)
    with open(p, "w", encoding="utf-8") as f:
        for i, did in enumerate(ids):
            rec = {"id": did, "text": texts[i], "ts": ts[i], "tags": tags[i], "thread_len": thread_len[i]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def cmd_build(args):
    cfg = AppConfig(provider=args.provider, model=args.model, dim=args.dim)
    provider = build_provider(cfg)
    if not provider.is_available():
        print(f"[mem-index] provider {args.provider} not available (check API key/env).", file=sys.stderr)
        sys.exit(2)

    texts = []
    if args.dir:
        texts.extend(list(_read_texts_from_dir(args.dir)))
    if args.jsonl:
        texts.extend(list(_read_texts_from_jsonl(args.jsonl)))
    if not texts:
        print("[mem-index] no input texts", file=sys.stderr)
        sys.exit(1)

    cache = EmbeddingCache(args.cache)
    ids, vecs = _ensure_vectors(texts, provider, cache, args.provider)
    os.makedirs(args.out, exist_ok=True)
    spec = IndexSpec(provider=args.provider, model=cfg.model, dim=cfg.dim, normalize=cfg.normalize)
    idx = FAISSIndex(os.path.join(args.out, "faiss.index"), spec)
    idx.load_or_build(vecs, ids)
    
    # Persist corpus metadata and IDs for search re-ranking
    _write_corpus(args.out, ids, texts)
    (pathlib.Path(args.out) / "ids.json").write_text(json.dumps(ids))
    
    print(f"[mem-index] built {len(ids)} vectors → {args.out}")

def cmd_search(args):
    from .index.faiss_index import FAISSIndex, IndexSpec
    import faiss
    import numpy as np

    cfg = AppConfig(provider=args.provider, model=args.model, dim=args.dim)
    provider = build_provider(cfg)
    if not provider.is_available():
        print(f"[mem-index] provider {args.provider} not available", file=sys.stderr)
        sys.exit(2)

    cache = EmbeddingCache(args.cache)
    q_vec = provider.embed_query(args.query)
    if provider.cfg.normalize:
        q = np.array([q_vec], dtype="float32")
    else:
        q = np.array([q_vec], dtype="float32")
    spec = IndexSpec(provider=args.provider, model=cfg.model, dim=cfg.dim, normalize=cfg.normalize)
    index_path = os.path.join(args.out, "faiss.index")
    if not os.path.exists(index_path):
        print(f"[mem-index] index not found at {index_path}", file=sys.stderr)
        sys.exit(1)
    index = faiss.read_index(index_path)
    D, I = index.search(q, max(32, args.k))  # over-fetch for better re-rank
    
    # Load corpus metadata and IDs for re-ranking
    ids_list_path = os.path.join(args.out, "ids.json")
    ids_list = json.loads(open(ids_list_path, "r", encoding="utf-8").read()) if os.path.exists(ids_list_path) else []
    
    corpus_path = os.path.join(args.out, "corpus.jsonl")
    meta = {}
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                meta[obj["id"]] = obj
    
    # Re-rank using semantic×temporal×salience blending
    now = time.time()
    pairs = []
    
    def days_old_for(doc_id: str) -> float:
        if doc_id in meta:
            return max(0.0, (now - float(meta[doc_id].get("ts", now))) / 86400.0)
        return 0.0
    
    def tags_for(doc_id: str) -> int:
        return int(meta.get(doc_id, {}).get("tags", 0))
    
    def thread_len_for(doc_id: str) -> int:
        return int(meta.get(doc_id, {}).get("thread_len", 1))
    
    for local_idx, sem in zip(I[0].tolist(), D[0].tolist()):
        if 0 <= local_idx < len(ids_list):
            doc_id = ids_list[local_idx]
            score = blend(
                semantic=float(sem),
                days_old=days_old_for(doc_id),
                tags=tags_for(doc_id),
                thread_len=thread_len_for(doc_id)
            )
            pairs.append((doc_id, score))
    
    # Sort by blended score and return top-k
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:args.k]
    
    # Return results with metadata
    results = []
    for doc_id, score in top:
        results.append({
            "id": doc_id, 
            "score": score, 
            "text": meta.get(doc_id, {}).get("text")
        })
    
    if hasattr(args, 'no_rerank') and args.no_rerank:
        # Return raw FAISS ranking for debugging
        print(json.dumps({"scores": D[0].tolist(), "ids": I[0].tolist()}))
    else:
        print(json.dumps({"results": results}))

def main(argv=None):
    p = argparse.ArgumentParser("mem-index")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build FAISS index from a directory and/or JSONL")
    pb.add_argument("--dir", type=str, help="Directory of .txt/.md/.json")
    pb.add_argument("--jsonl", type=str, help="JSONL with {'text': ...}")
    pb.add_argument("--out", type=str, default=".index")
    pb.add_argument("--cache", type=str, default=".cache/embeddings.sqlite3")
    pb.add_argument("--provider", type=str, default=os.getenv("EMBED_PROVIDER","voyage"))
    pb.add_argument("--model", type=str, default=os.getenv("EMBED_MODEL","voyage-2"))
    pb.add_argument("--dim", type=int, default=int(os.getenv("EMBED_DIM","1536")))
    pb.set_defaults(func=cmd_build)

    ps = sub.add_parser("search", help="Search an existing FAISS index")
    ps.add_argument("--out", type=str, default=".index")
    ps.add_argument("--cache", type=str, default=".cache/embeddings.sqlite3")
    ps.add_argument("--provider", type=str, default=os.getenv("EMBED_PROVIDER","voyage"))
    ps.add_argument("--model", type=str, default=os.getenv("EMBED_MODEL","voyage-2"))
    ps.add_argument("--dim", type=int, default=int(os.getenv("EMBED_DIM","1536")))
    ps.add_argument("-k", type=int, default=8)
    ps.add_argument("--no-rerank", action="store_true", help="Return raw FAISS ranking only (debug)")
    ps.add_argument("query", type=str)
    ps.set_defaults(func=cmd_search)

    args = p.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()