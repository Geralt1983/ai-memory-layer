from __future__ import annotations
import csv
import math
import os
import time
from typing import Iterable, List, Optional, Tuple

# Env toggle: set EMBED_AB_LOG to a file path (e.g., ./data/ab_metrics.csv)
# When present, DualWriteEmbeddings will append CSV rows.
ENV_LOG_PATH = "EMBED_AB_LOG"

CSV_HEADERS = [
    "ts_epoch_ms",
    "batch_size",
    "primary_provider",
    "primary_model",
    "shadow_provider",
    "shadow_model",
    "primary_ms",
    "shadow_ms",
    "mean_cosine",
    "shadow_error",     # empty if ok
]

def _now_ms() -> int:
    return int(time.time() * 1000)

def _cosine(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    # simple cosine without numpy
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (math.sqrt(na) * math.sqrt(nb))

def _cosine_batch(primary_vecs: List[List[float]], shadow_vecs: List[List[float]]) -> Optional[float]:
    if not primary_vecs or not shadow_vecs or len(primary_vecs) != len(shadow_vecs):
        return None
    cosines: List[float] = []
    for a, b in zip(primary_vecs, shadow_vecs):
        c = _cosine(a, b)
        if c is not None:
            cosines.append(c)
    if not cosines:
        return None
    return sum(cosines) / len(cosines)

def log_ab_event_row(
    path: str,
    batch_size: int,
    primary_provider: str,
    primary_model: str,
    shadow_provider: str,
    shadow_model: str,
    primary_ms: float,
    shadow_ms: Optional[float],
    mean_cosine: Optional[float],
    shadow_error: str = "",
) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(CSV_HEADERS)
        w.writerow([
            _now_ms(),
            batch_size,
            primary_provider,
            primary_model,
            shadow_provider,
            shadow_model,
            round(primary_ms, 3),
            round(shadow_ms, 3) if shadow_ms is not None else "",
            round(mean_cosine, 6) if mean_cosine is not None else "",
            shadow_error,
        ])

def ab_log_enabled() -> Optional[str]:
    path = os.getenv(ENV_LOG_PATH, "").strip()
    return path or None

def compute_cosine_mean(primary_vecs: List[List[float]], shadow_vecs: List[List[float]]) -> Optional[float]:
    return _cosine_batch(primary_vecs, shadow_vecs)