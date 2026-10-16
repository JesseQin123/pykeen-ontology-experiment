"""One ranking protocol for every method, so embeddings and graph-search baselines are compared like for like.

Filtered setting, both prediction directions, *realistic* rank for ties (mean of optimistic and pessimistic),
which matters a lot for sparse rule scores where many candidates tie at zero.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .data import KG, as_queries

# score_fn(heads, directions) -> (B, num_entities) float array; direction >= R means inverse relation
ScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def ranks(kg: KG, score_fn: ScoreFn, triples: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Filtered realistic rank of the true answer for each query; returns ranks for [tail queries, head queries]."""
    queries = as_queries(triples, kg.num_relations)
    known = kg.known_tails()
    out = np.empty(len(queries), dtype=np.float64)
    for s in range(0, len(queries), batch_size):
        q = queries[s : s + batch_size]
        scores = np.asarray(score_fn(q[:, 0], q[:, 1]), dtype=np.float64)
        for i, (h, d, t) in enumerate(q):
            row = scores[i]
            true = row[t]
            filt = [x for x in known[(int(h), int(d))] if x != t]
            row[filt] = -np.inf
            greater = np.count_nonzero(row > true)
            equal = np.count_nonzero(row == true) - 1
            out[s + i] = greater + 1 + equal / 2.0
    return out


def summarize(r: np.ndarray) -> dict[str, float]:
    return {
        "MRR": float(np.mean(1.0 / r)),
        "Hits@1": float(np.mean(r <= 1)),
        "Hits@3": float(np.mean(r <= 3)),
        "Hits@10": float(np.mean(r <= 10)),
        "MR": float(np.mean(r)),
        "n": int(len(r)),
    }
