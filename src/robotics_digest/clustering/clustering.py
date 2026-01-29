# robotics_digest/clustering.py
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

from ..models.models import Message


def cluster_relevant_period(
    messages: List[Message], 
    embeddings: np.ndarray, 
    start_day: int, 
    days: int = 14, 
    n_clusters: int = 12
) -> Dict[int, List[int]]:
    """Cluster messages over 14-day window to find stable topics."""
    base_ts = messages[0].ts
    window_idxs = [
        i for i, m in enumerate(messages)
        if start_day <= (m.ts - base_ts).days < start_day + days
    ]
    
    if len(window_idxs) < 50:  # Need enough data
        return {}
    
    window_embs = embeddings[window_idxs]
    clusters = cluster_messages(
        [messages[i] for i in window_idxs], 
        window_embs, 
        n_clusters=n_clusters
    )
    
    # Map back to global indices
    global_clusters = {}
    for cid, local_idxs in clusters.items():
        global_clusters[cid] = [window_idxs[i] for i in local_idxs]
    
    return global_clusters



def day_filter(messages: List[Message], day: int) -> List[int]:
    base = datetime(2025, 1, 1)
    return [
        i for i, m in enumerate(messages)
        if (m.ts - base).days == day
    ]

def cluster_for_day(
    day: int,
    messages: List[Message],
    embeddings: np.ndarray,
    n_clusters: int = 6,
) -> List[int]:
    idxs = day_filter(messages, day)
    if not idxs:
        return []
    day_msgs = [messages[i] for i in idxs]
    day_embs = embeddings[idxs, :]
    clusters = cluster_messages(day_msgs, day_embs, n_clusters=n_clusters)
    rep_local = select_representatives(day_msgs, day_embs, clusters)
    return [idxs[i] for i in rep_local]  # map back to global indices


def cluster_messages(
    messages: List[Message],
    embeddings: np.ndarray,
    n_clusters: int = 8,
) -> Dict[int, List[int]]:
    if len(messages) <= n_clusters:
        return {i: [i] for i in range(len(messages))}
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[int(label)].append(idx)
    return clusters

def select_representatives(
    messages: List[Message],
    embeddings: np.ndarray,
    clusters: Dict[int, List[int]],
    max_per_cluster: int = 3,
) -> List[int]:
    # simple scoring: decisions/risks/blockers first, then reaction count
    scores = []
    for i, m in enumerate(messages):
        score = 0
        if m.is_decision:
            score += 3
        if m.is_risk:
            score += 2
        if m.is_blocker:
            score += 2
        score += len(m.reactions)
        scores.append(score)

    selected: List[int] = []
    for label, idxs in clusters.items():
        idxs_sorted = sorted(idxs, key=lambda i: scores[i], reverse=True)
        selected.extend(idxs_sorted[:max_per_cluster])
    return selected
