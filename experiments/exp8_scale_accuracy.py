"""Experiment 8 - Does accuracy (not just latency) change as the graph gets bigger?

YAGO3-10 (123k entities, 1.08M training triples). We grow connected sub-graphs by BFS from random seeds
(keeps local density roughly constant, so "bigger" means more entities and facts, not just denser), and at each
size train RotatE and the path-rule learner on the sub-graph's training triples, then evaluate on test triples
inside it. Records accuracy, training time, and rule-mining time/size.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from pykeen.triples import TriplesFactory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import baselines, evaluate, kge  # noqa: E402
from kgv.data import KG  # noqa: E402

RESULTS = ROOT / "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_yago():
    from pykeen.datasets import YAGO310
    ds = YAGO310()
    tf = ds.training
    ent = [None] * tf.num_entities
    for k, v in tf.entity_to_id.items():
        ent[v] = k
    rel = [None] * tf.num_relations
    for k, v in tf.relation_to_id.items():
        rel[v] = k
    return (tf.mapped_triples.numpy(), ds.validation.mapped_triples.numpy(), ds.testing.mapped_triples.numpy(),
            ent, rel)


def ball(n_ent, train, size, seed_entities, seed):
    """Nested node sets: start from the fixed test entities, grow breadth-first (random order) to `size` nodes."""
    A = sp.csr_matrix((np.ones(len(train)), (train[:, 0], train[:, 2])), shape=(n_ent, n_ent))
    A = ((A + A.T) > 0).tocsr()
    rng = np.random.default_rng(seed)
    mark = np.zeros(n_ent, bool)
    frontier = np.unique(seed_entities)
    mark[frontier] = True
    count = len(frontier)
    while count < size and len(frontier):
        nb = np.unique(A[frontier].indices)
        nb = nb[~mark[nb]]
        rng.shuffle(nb)
        nb = nb[: size - count]
        mark[nb] = True
        count += len(nb)
        frontier = nb
    return mark


def sub_kg(train, valid, test, ent, rel, mask):
    keep = lambda T: T[mask[T[:, 0]] & mask[T[:, 2]]]
    tr, va, te = keep(train), keep(valid), keep(test)
    ids = np.unique(tr[:, [0, 2]])
    remap = -np.ones(len(ent), np.int64); remap[ids] = np.arange(len(ids))
    rmap = lambda T: np.stack([remap[T[:, 0]], T[:, 1], remap[T[:, 2]]], 1)
    va = va[(remap[va[:, 0]] >= 0) & (remap[va[:, 2]] >= 0)]
    te = te[(remap[te[:, 0]] >= 0) & (remap[te[:, 2]] >= 0)]
    tr, va, te = rmap(tr), rmap(va), rmap(te)
    tf = TriplesFactory(mapped_triples=torch.as_tensor(tr), entity_to_id={ent[i]: j for j, i in enumerate(ids)},
                        relation_to_id={r: i for i, r in enumerate(rel)})
    return KG("yago-sub", len(ids), len(rel), tr, va, te, [ent[i] for i in ids], list(rel), factory=tf)


def main():
    train, valid, test, ent, rel = load_yago()
    log("YAGO3-10", len(ent), len(train))
    res = []
    rng = np.random.default_rng(0)
    fixed = test[rng.permutation(len(test))[:1000]]  # the same 1,000 test triples at every size
    for size in (5_000, 15_000, 40_000, len(ent)):
        mask = ball(len(ent), train, size, fixed[:, [0, 2]].ravel(), seed=0) if size < len(ent) else np.ones(len(ent), bool)
        kg = sub_kg(train, valid, fixed, ent, rel, mask)
        te = kg.test
        row = {"entities": kg.num_entities, "train_triples": len(kg.train), "test_eval": len(te),
               "avg_degree": 2 * len(kg.train) / kg.num_entities}
        t = time.time()
        rules = baselines.PathRules(kg, samples_per_direction=500, candidates_per_direction=60, max_body=5e7)
        row["rules_build_s"] = time.time() - t
        row["rules_nnz_millions"] = sum(S.nnz for S in rules.S) / 1e6
        row["PathRules"] = evaluate.summarize(evaluate.ranks(kg, rules.score, te))
        del rules
        epochs = 40 if len(kg.train) < 400_000 else 25
        m, info = kge.train(kg, "RotatE", seed=0, device="mps", epochs=epochs, lr=1e-2)
        row["RotatE_train_s"] = info["train_seconds"]; row["RotatE_epochs"] = epochs
        row["RotatE"] = evaluate.summarize(evaluate.ranks(kg, kge.score_fn(m, kg.num_relations), te))
        log(row)
        res.append(row)
        (RESULTS / "exp8_scale_accuracy_yago.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
