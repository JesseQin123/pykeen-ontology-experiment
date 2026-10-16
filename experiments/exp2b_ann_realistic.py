"""Experiment 2b - ANN recall on realistic vectors.

Exp 2a used isotropic Gaussian vectors, the worst case for HNSW (no structure to exploit), so its recall numbers
understate real-world ANN quality. Here: (1) the real trained WN18RR RotatE entity embeddings, and (2) clustered
synthetic vectors (Gaussian mixture), at several efSearch settings.
Run as two processes (export with torch, search with faiss) - the two libomp runtimes deadlock in one process on macOS.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "runs/wn18rr_rotate_entities.npy"


def export():
    sys.path.insert(0, str(ROOT))
    import torch
    from kgv import kge
    m = torch.load(ROOT / "runs/wn18rr_RotatE_s0.pt", weights_only=False, map_location="cpu")
    np.save(EMB, kge.entity_embeddings(m))


def recall(X, Q, efs=(32, 64, 256), k=10):
    import faiss
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    gt = np.argsort(-(Q @ X.T), axis=1)[:, :k]
    t = time.perf_counter()
    idx = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200
    idx.add(X)
    out = {"N": len(X), "d": X.shape[1], "build_s": time.perf_counter() - t}
    for ef in efs:
        idx.hnsw.efSearch = ef
        t = time.perf_counter()
        _, I = idx.search(Q, k)
        out[f"ef{ef}_ms_per_query"] = (time.perf_counter() - t) * 1000 / len(Q)
        out[f"ef{ef}_recall@10"] = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(I, gt)]))
    return out


def search():
    rng = np.random.default_rng(0)
    res = {}
    X = np.load(EMB).astype(np.float32)
    res["real_WN18RR_RotatE"] = recall(X, X[rng.choice(len(X), 500, replace=False)])
    print(res, flush=True)
    for n in (100_000, 1_000_000):
        c = rng.standard_normal((n // 100, 256)).astype(np.float32)
        X = c[rng.integers(0, len(c), n)] + 0.35 * rng.standard_normal((n, 256)).astype(np.float32)
        res[f"clustered_synthetic_{n}"] = recall(X, X[rng.choice(n, 300, replace=False)])
        print(res, flush=True)
    (ROOT / "results/exp2b_ann.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["search"]:
        search()
    else:
        export()
        subprocess.run([sys.executable, __file__, "search"], check=True)
