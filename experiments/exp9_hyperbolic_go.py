"""Experiment 9 - Hyperbolic (Poincare) embeddings for the GO is_a hierarchy.

Tree-like hierarchies embed with low distortion in hyperbolic space (Nickel & Kiela 2017). Do they predict the
is_a links that GO curators added between 2020 and 2026 better than RotatE / text? Trained on the transitive
closure of the 2020 is_a graph (as in the original paper), scored with the paper's
is-a score  -(1 + alpha * (||parent|| - ||child||)) * d(child, parent).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import evaluate, go  # noqa: E402

RESULTS = ROOT / "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main(old="2020-01-01", new="2026-08-05", dim=50, epochs=50, alphas=(0.0, 1.0, 10.0, 100.0, 1000.0)):
    from gensim.models.poincare import PoincareModel
    kg, info, kinds, _, terms = go.temporal(old, new)
    isa = kg.relation_labels.index("is_a")
    lab = kg.entity_labels
    edges = [(lab[h], lab[t]) for h, r, t in kg.train if r == isa]
    anc = go.closure_is_a([(a, "is_a", b) for a, b in edges])
    closure = sorted({(c, a) for c, ancs in anc.items() for a in ancs})
    log("is_a edges", len(edges), "closure", len(closure))
    t = time.time()
    path = ROOT / "runs" / f"poincare_go_{old}_d{dim}_e{epochs}.model"
    if path.exists():
        model = PoincareModel.load(str(path))
    else:
        model = PoincareModel(closure, size=dim, negative=10, burn_in=10, workers=1, seed=0)
        model.train(epochs=epochs, batch_size=512, print_every=10_000)
        model.save(str(path))
    train_s = time.time() - t
    log("trained", train_s)
    E = np.zeros((kg.num_entities, dim))
    for i, e in enumerate(lab):
        if e in model.kv:
            E[i] = model.kv[e]
    norm = np.linalg.norm(E, axis=1)

    def pdist(u, V):
        uu, vv = (u ** 2).sum(), (V ** 2).sum(1)
        diff = ((V - u) ** 2).sum(1)
        return np.arccosh(1 + 2 * diff / np.maximum((1 - uu) * (1 - vv), 1e-9))

    def make_score(alpha):
        def score(heads, dirs):
            out = np.full((len(heads), kg.num_entities), -1e9)
            for i, (h, d) in enumerate(zip(heads, dirs)):
                dist = pdist(E[h], E)
                if d == isa:  # predict parent: parent should have smaller norm
                    out[i] = -(1 + alpha * (norm - norm[h])) * dist
                elif d == isa + kg.num_relations:  # predict child
                    out[i] = -(1 + alpha * (norm[h] - norm)) * dist
                else:
                    out[i] = -dist
                out[i, h] = -1e9
            return out
        return score

    mask = kg.test[:, 1] == isa
    test = kg.test[mask]
    k = kinds[mask]
    res = {"setup": {"is_a_edges": len(edges), "closure_pairs": len(closure), "dim": dim, "epochs": epochs,
                     "train_seconds": train_s, "mean_norm": float(norm.mean())}, "by_alpha": {}}
    for alpha in alphas:
        r = evaluate.ranks(kg, make_score(alpha), test)
        tail = r[: len(test)]
        res["by_alpha"][str(alpha)] = {"all_tail_only": evaluate.summarize(tail),
                                       "novel_tail_only": evaluate.summarize(tail[k == "novel"]),
                                       "entailed_tail_only": evaluate.summarize(tail[k == "entailed"])}
        log(alpha, res["by_alpha"][str(alpha)]["all_tail_only"])
        (RESULTS / "exp9_poincare_go.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
