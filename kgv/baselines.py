"""Non-embedding baselines: what plain graph statistics / graph search can already find.

* Popularity: the most frequent tails of a relation (a "degree prior").
* PathRules: an AnyBURL-style rule learner restricted to paths of length <= 2.
  It mines path patterns that co-occur with a relation (e.g. hypernym <- hypernym . hypernym,
  or similar_to <- similar_to^-1), measures each pattern's confidence on the training graph, and scores
  a candidate by noisy-or over the patterns that connect it to the query entity.
  This is the honest stand-in for "graph search": every prediction comes with an explicit path.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

from .data import KG


def direction_matrices(kg: KG, triples: np.ndarray | None = None) -> list[sp.csr_matrix]:
    triples = kg.train if triples is None else triples
    n, R = kg.num_entities, kg.num_relations
    mats = []
    for r in range(R):
        m = triples[triples[:, 1] == r]
        a = sp.csr_matrix((np.ones(len(m), np.float32), (m[:, 0], m[:, 2])), shape=(n, n))
        a.data[:] = 1.0
        mats.append(a)
    return mats + [a.T.tocsr() for a in mats]


class Popularity:
    def __init__(self, kg: KG):
        self.pop = np.stack([np.asarray(a.sum(0)).ravel() for a in direction_matrices(kg)])

    def score(self, heads: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        return self.pop[dirs]


class PathRules:
    def __init__(
        self,
        kg: KG,
        samples_per_direction: int = 1000,
        candidates_per_direction: int = 120,
        top_patterns: int = 40,
        min_support: int = 2,
        pseudo_count: float = 5.0,
        max_body: float = 3e7,
        seed: int = 0,
    ):
        self.kg = kg
        rng = random.Random(seed)
        A = direction_matrices(kg)
        D = len(A)
        n = kg.num_entities
        out = defaultdict(list)  # x -> [(d, y)]
        pair_dirs = defaultdict(list)  # (x, y) -> [d]
        for d, a in enumerate(A):
            coo = a.tocoo()
            for x, y in zip(coo.row.tolist(), coo.col.tolist()):
                out[x].append((d, y))
                pair_dirs[(x, y)].append(d)
        indeg = [np.asarray(a.sum(0)).ravel() for a in A]
        outdeg = [np.asarray(a.sum(1)).ravel() for a in A]

        self.rules: dict[int, list[tuple[tuple[int, ...], float, int, int]]] = {}
        self.S: list[sp.csr_matrix] = []
        pop = np.stack([indeg[d] for d in range(D)])
        for q in range(D):
            coo = A[q].tocoo()
            edges = list(zip(coo.row.tolist(), coo.col.tolist()))
            sample = rng.sample(edges, min(samples_per_direction, len(edges)))
            counts: Counter = Counter()
            for x, y in sample:
                for d in pair_dirs[(x, y)]:
                    if d != q:
                        counts[(d,)] += 1
                for d1, z in out[x]:
                    if z == y or z == x:
                        continue
                    for d2 in pair_dirs.get((z, y), ()):
                        counts[(d1, d2)] += 1
            scored = []
            for pat, _ in counts.most_common(candidates_per_direction):
                if len(pat) == 1:
                    M = A[pat[0]]
                else:
                    if float(indeg[pat[0]] @ outdeg[pat[1]]) > max_body:
                        continue  # hub-explosion pattern (e.g. "same nationality"); skip like AnyBURL would time out
                    M = (A[pat[0]] @ A[pat[1]]).tocsr()
                    M.setdiag(0)
                    M.eliminate_zeros()
                    M.data[:] = 1.0
                body = M.nnz
                support = int(M.multiply(A[q]).nnz)
                if support < min_support:
                    continue
                conf = support / (body + pseudo_count)
                scored.append((pat, conf, support, body, M))
            scored.sort(key=lambda x: -x[1])
            scored = scored[:top_patterns]
            self.rules[q] = [(p, c, s, b) for p, c, s, b, _ in scored]
            S = sp.csr_matrix((n, n), dtype=np.float64)
            for _, c, _, _, M in scored:
                S = S + M.astype(np.float64) * (-np.log1p(-min(c, 0.999)))
            self.S.append(S.tocsr())
        # tiny popularity prior breaks ties among candidates no rule reaches
        self.prior = 1e-6 * pop / np.maximum(pop.max(1, keepdims=True), 1)

    def score(self, heads: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        out = np.empty((len(heads), self.kg.num_entities))
        for i, (h, d) in enumerate(zip(heads, dirs)):
            out[i] = self.S[d][h].toarray().ravel() + self.prior[d]
        return out

    def reachable(self, heads: np.ndarray, dirs: np.ndarray, tails: np.ndarray) -> np.ndarray:
        """True if any learned rule connects head to tail (i.e. graph search could have found it)."""
        return np.array([self.S[d][h, t] > 0 for h, d, t in zip(heads, dirs, tails)])

    def describe(self, k: int = 5) -> dict[str, list[str]]:
        R = self.kg.num_relations
        name = lambda d: self.kg.relation_labels[d % R] + ("^-1" if d >= R else "")
        return {
            name(q): [f"{' . '.join(name(d) for d in p)}  conf={c:.2f} supp={s}" for p, c, s, _ in rules[:k]]
            for q, rules in self.rules.items()
        }
