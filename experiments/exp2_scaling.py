"""Experiment 2 - Where do embeddings help performance on large graphs, and what do they cost?

(a) Query cost vs graph size. "Find the 10 most related concepts" answered two ways:
      * embedding nearest neighbours: brute force and an HNSW index (FAISS)
      * graph traversal: k-hop neighbourhood expansion on a power-law graph of the same size
    They answer *different* questions (similarity vs connectivity) - the point is the cost profile:
    embeddings cost O(N d) brute force or ~O(log N) with ANN, independent of hops and hubs;
    traversal cost depends on degree^k and explodes at hubs.
(b) Training cost vs graph size: measured RotATE epoch time on growing subsets of FB15k-237, plus
    the memory a model would need at scale.
(c) On the real WN18RR graph: latency of (h, r, ?) answering with the trained model vs the path-rule index.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import baselines, data, kge  # noqa: E402

RESULTS = ROOT / "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chung_lu(n, avg_deg=10, gamma=2.2, seed=0):
    """Power-law random graph (expected degrees ~ Zipf) - mimics hubs like 'person' or 'United States'."""
    rng = np.random.default_rng(seed)
    w = (np.arange(1, n + 1) ** (-1 / (gamma - 1))).astype(np.float64)
    w *= avg_deg * n / w.sum()
    m = int(avg_deg * n / 2)
    p = w / w.sum()
    a = rng.choice(n, m, p=p)
    b = rng.choice(n, m, p=p)
    g = sp.csr_matrix((np.ones(m, np.float32), (a, b)), shape=(n, n))
    return ((g + g.T) > 0).astype(np.float32).tocsr()


def khop(g, src, k):
    frontier = np.array([src])
    seen = np.zeros(g.shape[0], bool)
    seen[src] = True
    for _ in range(k):
        nb = np.unique(g[frontier].indices)
        nb = nb[~seen[nb]]
        seen[nb] = True
        frontier = nb
        if len(frontier) == 0:
            break
    return int(seen.sum())


def part_a(sizes=(10_000, 100_000, 1_000_000, 3_000_000), d=256, queries=200):
    import faiss  # imported lazily: faiss + torch each ship libomp and can deadlock in one process on macOS

    out = []
    rng = np.random.default_rng(0)
    for n in sizes:
        row = {"N": n}
        X = rng.standard_normal((n, d), dtype=np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Q = X[rng.choice(n, queries, replace=False)]
        t = time.perf_counter()
        D = Q @ X.T
        bf = np.argpartition(-D, 10, axis=1)[:, :10]
        row["bruteforce_ms_per_query"] = (time.perf_counter() - t) * 1000 / queries
        del D
        t = time.perf_counter()
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.add(X)
        row["hnsw_build_s"] = time.perf_counter() - t
        index.hnsw.efSearch = 64
        t = time.perf_counter()
        _, I = index.search(Q, 10)
        row["hnsw_ms_per_query"] = (time.perf_counter() - t) * 1000 / queries
        row["hnsw_recall@10"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(I, bf)]))
        row["embedding_memory_GB(float32)"] = n * d * 4 / 1e9
        del index, X

        g = chung_lu(n)
        deg = np.diff(g.indptr)
        srcs = np.concatenate([rng.choice(n, queries // 2), np.argsort(-deg)[: queries // 20]])  # random + hubs
        for k in (2, 3):
            times, touched = [], []
            for s in srcs:
                t = time.perf_counter()
                touched.append(khop(g, s, k))
                times.append((time.perf_counter() - t) * 1000)
            times, touched = np.array(times), np.array(touched)
            row[f"{k}hop_ms_p50"] = float(np.percentile(times, 50))
            row[f"{k}hop_ms_p99"] = float(np.percentile(times, 99))
            row[f"{k}hop_nodes_touched_p50"] = int(np.percentile(touched, 50))
            row[f"{k}hop_nodes_touched_max_frac"] = float(touched.max() / n)
        out.append(row)
        log(row)
    return out


def part_b(device="mps"):
    kg = data.load("fb15k237")
    from pykeen.triples import TriplesFactory
    rows = []
    rng = np.random.default_rng(0)
    for frac in (0.25, 0.5, 1.0):
        idx = rng.permutation(len(kg.train))[: int(frac * len(kg.train))]
        tf = TriplesFactory(mapped_triples=kg.factory.mapped_triples[idx], entity_to_id=kg.factory.entity_to_id,
                            relation_to_id=kg.factory.relation_to_id)
        _, info = kge.train(kg, "RotatE", tf=tf, device=device, epochs=2, lr=1e-2)
        rows.append({"triples": len(idx), "epoch_seconds": info["train_seconds"] / 2})
        log(rows[-1])
    per_m = np.mean([r["epoch_seconds"] / r["triples"] for r in rows]) * 1e6
    proj = []
    for ents, trips in ((1e6, 1e7), (1e7, 1e8), (1e8, 1e9)):
        params_gb = ents * 256 * 2 * 4 / 1e9  # complex dim 256 = 512 floats
        proj.append({"entities": ents, "triples": trips,
                     "epoch_hours_single_M5Pro(linear extrapolation)": per_m * trips / 1e6 / 3600,
                     "entity_params_GB": params_gb, "with_Adam_state_GB": params_gb * 3})
    return {"measured": rows, "seconds_per_million_triples_per_epoch": per_m, "projection": proj}


def part_c():
    kg = data.load("wn18rr")
    rules = baselines.PathRules(kg)
    import torch
    model = torch.load(ROOT / "runs/wn18rr_RotatE_s0.pt", weights_only=False, map_location="mps")
    f = kge.score_fn(model, kg.num_relations)
    q = data.as_queries(kg.test, kg.num_relations)[:2000]
    out = {}
    for name, fn in (("RotatE (1-vs-all, mps)", f), ("PathRules (sparse rows)", rules.score)):
        fn(q[:32, 0], q[:32, 1])
        t = time.perf_counter()
        for s in range(0, len(q), 32):
            sc = fn(q[s : s + 32, 0], q[s : s + 32, 1])
            np.argpartition(-sc, 10, axis=1)
        out[name] = (time.perf_counter() - t) * 1000 / len(q)
    return {"ms_per_query_top10": out}


if __name__ == "__main__":
    out = RESULTS / "exp2_scaling.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    parts = sys.argv[1] if len(sys.argv) > 1 else "abc"
    if "a" in parts:
        res["a_query_cost_vs_size"] = part_a()
        out.write_text(json.dumps(res, indent=2))
    if "b" not in parts:
        sys.exit(0)
    res["b_training_cost"] = part_b()
    (RESULTS / "exp2_scaling.json").write_text(json.dumps(res, indent=2))
    res["c_real_graph_latency"] = part_c()
    (RESULTS / "exp2_scaling.json").write_text(json.dumps(res, indent=2))
    log("done")
