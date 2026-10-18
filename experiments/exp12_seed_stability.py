"""Experiment 12 - How stable are embedding suggestions across random seeds?

Three RotatE models that differ only in the random seed (identical data and hyper-parameters, near-identical MRR).
For every test query: overlap of their top-10 candidate lists, and agreement of the top-1 suggestion.
If suggestions are seed-dependent, any single run's "interesting suggestions" are partly noise.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import data, kge  # noqa: E402

kg = data.load("wn18rr")
q = data.as_queries(kg.test, kg.num_relations)
uq = np.unique(q[:, :2], axis=0)
known = kg.known_tails()
tops = {}
for s in (0, 1, 2):
    m = torch.load(ROOT / f"runs/wn18rr_RotatE_s{s}.pt", weights_only=False, map_location="cpu")
    f = kge.score_fn(m, kg.num_relations)
    T = []
    for i in range(0, len(uq), 256):
        b = uq[i : i + 256]
        sc = f(b[:, 0], b[:, 1]).astype(np.float64)
        for j, (h, d) in enumerate(b):
            sc[j, list(known.get((int(h), int(d)), ()))] = -np.inf  # suggest only what is not already known
            sc[j, h] = -np.inf
        T.append(np.argsort(-sc, axis=1)[:, :10])
    tops[s] = np.concatenate(T)
res = {"queries": int(len(uq))}
for a, b in combinations(tops, 2):
    ov = np.array([len(set(x) & set(y)) for x, y in zip(tops[a], tops[b])])
    res[f"seed{a}_vs_seed{b}"] = {"mean_top10_overlap": float(ov.mean()), "top1_agree": float(np.mean(tops[a][:, 0] == tops[b][:, 0])),
                                  "share_queries_overlap<=5": float(np.mean(ov <= 5))}
all3 = np.mean([len(set(x) & set(y) & set(z)) for x, y, z in zip(tops[0], tops[1], tops[2])])
res["mean_top10_in_all_three"] = float(all3)
print(json.dumps(res, indent=2))
(ROOT / "results/exp12_seed_stability.json").write_text(json.dumps(res, indent=2))
