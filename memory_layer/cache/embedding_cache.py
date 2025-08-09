from __future__ import annotations
import sqlite3, os, json, time, hashlib
from typing import Iterable, List, Tuple, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings(
  content_hash TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  normalize INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_provider ON embeddings(provider, model);
"""

def _db_path(path: Optional[str]) -> str:
    return path or os.getenv("EMBED_CACHE_PATH", ".cache/embeddings.sqlite3")

class EmbeddingCache:
    def __init__(self, path: Optional[str] = None):
        self.path = _db_path(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.cx = sqlite3.connect(self.path)
        self.cx.execute("PRAGMA journal_mode=WAL;")
        self.cx.executescript(SCHEMA)

    @staticmethod
    def content_hash(text: str) -> str:
        norm = " ".join(text.split()).strip()
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def get_many(self, keys: Iterable[str]) -> dict[str, List[float]]:
        q = "SELECT content_hash, vector FROM embeddings WHERE content_hash IN ({})"
        placeholders = ",".join("?" for _ in keys)
        if not placeholders:
            return {}
        cur = self.cx.execute(q.format(placeholders), list(keys))
        out = {}
        for k, blob in cur.fetchall():
            out[k] = json.loads(blob)
        return out

    def put_many(self, items: Iterable[Tuple[str, List[float], str, str, int, int]]) -> None:
        # (hash, vec, provider, model, dim, normalize)
        now = time.time()
        rows = [(h, json.dumps(v), p, m, d, n, now) for h, v, p, m, d, n in items]
        self.cx.executemany(
            "INSERT OR REPLACE INTO embeddings(content_hash, vector, provider, model, dim, normalize, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self.cx.commit()