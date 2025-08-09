from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ScoredMemory:
    content: str
    embedding: List[float]
    timestamp: float
    raw_similarity: float
    human_score: float
    metadata: dict

def human_recall_score(
    raw_similarity: float,
    timestamp: float,
    now: Optional[float] = None,
    tags: Optional[List[str]] = None,
) -> float:
    """Human-like scoring: high similarity + recency + salience bumps."""
    if now is None:
        now = time.time()
    
    # Temporal decay: recent memories get boosted
    hours_ago = max(0, now - timestamp) / 3600.0
    if hours_ago < 1:
        recency = 1.0
    elif hours_ago < 24:
        recency = 0.9
    elif hours_ago < 168:  # 1 week
        recency = 0.7
    else:
        recency = 0.5
    
    # Salience bump for important/emotional content
    salience = 1.0
    if tags:
        important_tags = {"error", "critical", "urgent", "decision", "milestone"}
        if any(tag.lower() in important_tags for tag in tags):
            salience = 1.2
    
    # Combine: similarity is primary, recency and salience are modifiers
    return min(1.0, raw_similarity * recency * salience)

def rank_memories(
    memories: List[tuple[str, List[float], float, dict]], 
    query_embedding: List[float],
    k: int = 8
) -> List[ScoredMemory]:
    """Rank memories by human-like scoring."""
    now = time.time()
    scored = []
    
    for content, embedding, timestamp, metadata in memories:
        # Cosine similarity
        dot = sum(a * b for a, b in zip(query_embedding, embedding))
        norm_a = math.sqrt(sum(x * x for x in query_embedding))
        norm_b = math.sqrt(sum(x * x for x in embedding))
        raw_sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
        
        # Human scoring
        tags = metadata.get("tags", [])
        human_score = human_recall_score(raw_sim, timestamp, now, tags)
        
        scored.append(ScoredMemory(
            content=content,
            embedding=embedding,
            timestamp=timestamp,
            raw_similarity=raw_sim,
            human_score=human_score,
            metadata=metadata
        ))
    
    # Sort by human score, then by raw similarity
    scored.sort(key=lambda x: (x.human_score, x.raw_similarity), reverse=True)
    return scored[:k]

def blend(semantic: float, days_old: float, tags: int = 0, thread_len: int = 1) -> float:
    """
    Blend semantic similarity with temporal decay and salience.
    
    Args:
        semantic: Raw semantic similarity score (0-1, higher = more similar)
        days_old: Age of content in days (0 = today, higher = older)
        tags: Salience indicator (0 = normal, 1+ = important/critical)
        thread_len: Thread/conversation length (1 = standalone, higher = more context)
    
    Returns:
        Blended score optimized for human-like relevance
    """
    # Temporal decay: exponential decay with 7-day half-life
    temporal_factor = math.exp(-days_old * math.log(2) / 7.0)
    
    # Salience boost: critical/important content gets boosted
    salience_factor = 1.0 + (0.3 * min(tags, 3))  # Cap at 3x boost
    
    # Thread length boost: longer conversations often have more context
    thread_factor = 1.0 + (0.1 * math.log(max(1, thread_len)))
    
    # Combine with weighted importance:
    # - Semantic similarity: 60% base weight
    # - Temporal relevance: 25% weight
    # - Salience: 15% weight (multiplicative boost)
    base_score = (0.6 * semantic) + (0.25 * temporal_factor) + (0.15 * min(1.0, thread_factor))
    
    # Apply salience as multiplicative factor
    final_score = base_score * salience_factor
    
    # Ensure score stays in reasonable bounds
    return min(2.0, max(0.0, final_score))