"""Experiment 7 - Does the choice of embedding model matter on ontology graphs? A PyKEEN model zoo.

Same budget for every model (sLCWA, 128 negatives, self-adversarial NSSA loss, 40 epochs, best of two learning
rates picked on the WN18RR validation split), evaluated with the shared protocol. Models span translational
(TransE/H/D), bilinear (DistMult, ComplEx, SimplE, TuckER, AutoSF), rotational (RotatE, QuatE, PairRE),
box/region (BoxE), Euclidean multi-relational (MuRE), and others. Failures are recorded, not hidden.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import data, evaluate, kge  # noqa: E402

RESULTS, RUNS = ROOT / "results", ROOT / "runs"
MODELS = {
    "TransE": {}, "TransH": {}, "TransD": {}, "DistMult": {}, "ComplEx": {}, "SimplE": {},
    "RotatE": {}, "QuatE": {}, "PairRE": {}, "MuRE": {}, "BoxE": {}, "AutoSF": {}, "CrossE": {},
    "TuckER": dict(model_kwargs=dict(embedding_dim=200, relation_dim=200)),
    "ComplEx[softplus]": dict(_model="ComplEx", loss="SoftplusLoss", loss_kwargs={}),
    "DistMult[softplus]": dict(_model="DistMult", loss="SoftplusLoss", loss_kwargs={}),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    kg = data.load("wn18rr")
    rng = np.random.default_rng(0)
    val = kg.valid[rng.choice(len(kg.valid), 600, replace=False)]
    out_path = RESULTS / "exp7_model_zoo_wn18rr.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for label, ov in MODELS.items():
        if label in res:
            continue
        ov = dict(ov)
        name = ov.pop("_model", label)
        best = None
        for lr in (1e-2, 1e-3):
            try:
                m, info = kge.train(kg, name, seed=0, device="mps", lr=lr, **ov)
                f = kge.score_fn(m, kg.num_relations)
                v = evaluate.summarize(evaluate.ranks(kg, f, val))["MRR"]
                log(label, lr, f"val MRR={v:.3f}", f"{info['train_seconds']:.0f}s")
                if best is None or v > best[0]:
                    best = (v, lr, m, info)
            except Exception as ex:  # noqa: BLE001
                log(label, lr, "FAILED", repr(ex)[:200])
                traceback.print_exc()
        if best is None:
            res[label] = {"failed": True}
        else:
            v, lr, m, info = best
            r = evaluate.ranks(kg, kge.score_fn(m, kg.num_relations), kg.test)
            q = data.as_queries(kg.test, kg.num_relations)
            rel = q[:, 1] % kg.num_relations
            res[label] = {"lr": lr, "val_MRR": v, "train_seconds": info["train_seconds"], "test": evaluate.summarize(r),
                          "per_relation_MRR": {kg.relation_labels[i]: float(np.mean(1 / r[rel == i])) for i in range(kg.num_relations)}}
            log(label, "TEST", res[label]["test"])
        out_path.write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
