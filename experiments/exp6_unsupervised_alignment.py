"""Experiment 6 - The strongest fair test of "embeddings naturally align two taxonomies": ZERO seeds.

Independent embeddings + cosine is chance (exp 3). But maybe the *shapes* match even if the coordinates don't.
We try the best seed-free methods we know of:
  * Gromov-Wasserstein (GW) optimal transport on graph geometry (shortest-path distances) - matches nodes by
    how they sit relative to everything else, no coordinates or names needed
  * GW on the geometry of independently trained embedding spaces
  * GW followed by iterative Procrustes self-learning (MUSE-style refinement)
  * degree-profile matching (trivial structural baseline)
Datasets: induced subgraphs of the renamed WN18RR / FB15k-237 views (same construction as exp 3), and the real
OAEI Anatomy pair (mouse vs human), where the two ontologies were built independently.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import ot
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import data  # noqa: E402

RESULTS = ROOT / "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hits(T_or_S, gold_rows, gold_cols):
    S = T_or_S
    true = S[gold_rows, gold_cols]
    r = (S[gold_rows] > true[:, None]).sum(1) + 1
    return {"Hits@1": float(np.mean(r <= 1)), "Hits@10": float(np.mean(r <= 10)), "MRR": float(np.mean(1 / r)),
            "n": int(len(r)), "chance_Hits@1": 1 / S.shape[1]}


def adj(n, edges):
    e = np.asarray(edges)
    A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return ((A + A.T) > 0).astype(np.float64)


def sp_cost(A, cap=8):
    D = shortest_path(A, directed=False, unweighted=True)
    D[np.isinf(D)] = cap
    return np.minimum(D, cap) / cap


def gw(C1, C2, p, q, eps):
    return ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, "square_loss", epsilon=eps, max_iter=200, tol=1e-7)


def procrustes_refine(X, Y, S, rounds=5, k=500):
    """Self-learning: take the k most confident mutual matches as seeds, fit an orthogonal map, re-match."""
    for _ in range(rounds):
        i = S.argmax(1)
        mutual = np.where(S.argmax(0)[i] == np.arange(len(i)))[0]
        conf = S[mutual, i[mutual]]
        sel = mutual[np.argsort(-conf)[:k]]
        a, b = X[sel], Y[i[sel]]
        u, _, vt = np.linalg.svd(b.T @ a)
        Yw = Y @ (u @ vt)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        Yn = Yw / np.linalg.norm(Yw, axis=1, keepdims=True)
        S = Xn @ Yn.T
    return S


def degree_match(A, B):
    da, db = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    return -np.abs(np.log1p(da)[:, None] - np.log1p(db)[None, :]) + 1e-6 * np.random.default_rng(0).random((len(da), len(db)))


def spectral_embed(A, d=32):
    from scipy.sparse.linalg import eigsh
    deg = np.asarray(A.sum(1)).ravel() + 1e-9
    Dm = sp.diags(1 / np.sqrt(deg))
    L = sp.eye(A.shape[0]) - Dm @ A @ Dm
    w, v = eigsh(L, k=d + 1, sigma=-1e-3, which="LM")
    return np.abs(v[:, 1:])  # abs removes per-eigenvector sign ambiguity


def run_pair(name, nA, eA, nB, eB, gold, embA=None, embB=None):
    """gold: array of (a_idx, b_idx). Returns metrics for each seed-free method."""
    A, B = adj(nA, eA), adj(nB, eB)
    out = {"nodes_A": nA, "nodes_B": nB, "edges_A": len(eA), "edges_B": len(eB), "gold": len(gold)}
    ga, gb = gold[:, 0], gold[:, 1]
    p = np.asarray(A.sum(1)).ravel() + 1; p /= p.sum()
    q = np.asarray(B.sum(1)).ravel() + 1; q /= q.sum()
    out["degree profile"] = hits(degree_match(A, B), ga, gb)
    t = time.time()
    C1, C2 = sp_cost(A), sp_cost(B)
    best = None
    for eps in (5e-3, 1e-3, 5e-4):
        try:
            T = gw(C1, C2, p, q, eps)
        except Exception as ex:  # noqa: BLE001
            log(name, "gw failed", eps, repr(ex)[:80]); continue
        if not np.isfinite(T).all() or T.sum() == 0:
            continue
        m = hits(T, ga, gb)
        out[f"GW on graph shortest paths (eps={eps})"] = m
        if best is None or m["Hits@1"] > best[0]["Hits@1"]:
            best = (m, T, eps)
    out["gw_seconds"] = time.time() - t
    if best is not None:
        log(name, "GW best", best[2], best[0])
        XA, XB = spectral_embed(A), spectral_embed(B)
        out["GW + spectral Procrustes self-learning"] = hits(procrustes_refine(XA, XB, best[1] / best[1].max()), ga, gb)
    if embA is not None:
        D1 = ot.dist(embA, embA); D2 = ot.dist(embB, embB)
        D1 /= D1.max(); D2 /= D2.max()
        try:
            T2 = gw(D1, D2, p, q, 1e-3)
            out["GW on independent RotatE embedding geometry"] = hits(T2, ga, gb)
            out["GW(embeddings) + Procrustes self-learning"] = hits(procrustes_refine(embA, embB, T2 / T2.max()), ga, gb)
        except Exception as ex:  # noqa: BLE001
            out["GW on independent RotatE embedding geometry"] = repr(ex)[:120]
    return out


def views_subgraph(dsname, n_nodes=2500, seed=0):
    """Renamed views as in exp 3, restricted to a BFS ball so GW stays tractable."""
    sys.path.insert(0, str(ROOT / "experiments"))
    from exp3_alignment import make_views
    kg = data.load(dsname)
    A, B, gold, perm_e, _ = make_views(kg)
    full = adj(kg.num_entities, kg.train[:, [0, 2]])
    rng = np.random.default_rng(seed)
    deg = np.asarray(full.sum(1)).ravel()
    start = int(rng.choice(np.where(deg >= 5)[0]))
    seen, frontier = [start], [start]
    mark = np.zeros(kg.num_entities, bool); mark[start] = True
    while len(seen) < n_nodes and frontier:
        nb = np.unique(full[frontier].indices)
        nb = nb[~mark[nb]]
        rng.shuffle(nb)
        nb = nb[: n_nodes - len(seen)]
        mark[nb] = True
        seen += nb.tolist(); frontier = nb.tolist()
    keep = np.array(sorted(seen))
    ia = {e: i for i, e in enumerate(keep)}
    eA = [(ia[h], ia[t]) for h, _, t in A if h in ia and t in ia]
    eB_orig = [(h, t) for h, _, t in B if h in ia and t in ia]
    # B is renamed: shuffle node order so index identity leaks nothing
    perm = rng.permutation(len(keep))
    eB = [(perm[ia[h]], perm[ia[t]]) for h, t in eB_orig]
    nodesA = {x for e in eA for x in e}
    nodesB_orig = {ia[x] for e in eB_orig for x in e}
    g = np.array([(i, perm[i]) for i in range(len(keep)) if i in nodesA and i in nodesB_orig])
    return len(keep), eA, len(keep), eB, g


def anatomy():
    from kgv import oaei
    M, H = oaei.load("mouse"), oaei.load("human")
    mi = {c: i for i, c in enumerate(M.classes)}
    hi = {c: i for i, c in enumerate(H.classes)}
    eM = [(mi[h], mi[t]) for h, _, t in M.edges]
    eH = [(hi[h], hi[t]) for h, _, t in H.edges]
    g = np.array([(mi[a], hi[b]) for a, b in oaei.reference() if a in mi and b in hi])
    return len(M.classes), eM, len(H.classes), eH, g


def main():
    res = {}
    for ds in ("fb15k237", "wn18rr"):
        nA, eA, nB, eB, g = views_subgraph(ds)
        log(ds, "subgraph", nA, len(eA), len(eB), "gold", len(g))
        res[f"{ds} views (2,500-node ball)"] = run_pair(ds, nA, eA, nB, eB, g)
        log(ds, res[f"{ds} views (2,500-node ball)"])
        (RESULTS / "exp6_unsupervised_alignment.json").write_text(json.dumps(res, indent=2))
    nA, eA, nB, eB, g = anatomy()
    res["OAEI anatomy (mouse vs human, independent ontologies)"] = run_pair("anatomy", nA, eA, nB, eB, g)
    log("anatomy", res["OAEI anatomy (mouse vs human, independent ontologies)"])
    (RESULTS / "exp6_unsupervised_alignment.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
