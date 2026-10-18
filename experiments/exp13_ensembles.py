"""Experiment 13 - Best practical system: fuse rules (AnyBURL), embeddings, text and a reasoner.

Fusion by reciprocal-rank fusion (RRF, k=60): no weights tuned on the test set. Also a cascade
(AnyBURL candidates first, ordered by confidence; the embedding orders the rest).
WN18RR (random hold-out) and GO 2020 -> 2026 (prospective).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from kgv import anyburl, data, evaluate, go, kge  # noqa: E402

RESULTS = ROOT / "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def anyburl_scores(kg, preds):
    """(head, direction) -> {entity: confidence}, from AnyBURL's top-k lists for the test queries."""
    R = kg.num_relations
    d = {}
    for (h, r, t), (hc, tc) in preds.items():
        d.setdefault((h, r), {}).update(dict(tc))
        d.setdefault((t, r + R), {}).update(dict(hc))
    def f(heads, dirs):
        out = np.zeros((len(heads), kg.num_entities))
        for i, (h, dd) in enumerate(zip(heads, dirs)):
            for e, c in d.get((int(h), int(dd)), {}).items():
                out[i, e] = c
        return out
    return f


def ranks_matrix(s):
    """Per-row rank of every column (1 = best), ties averaged."""
    order = np.argsort(-s, axis=1, kind="stable")
    r = np.empty_like(order, dtype=np.float64)
    rows = np.arange(s.shape[0])[:, None]
    r[rows, order] = np.arange(1, s.shape[1] + 1)[None, :]
    return r


def rrf(*fns, k=60.0):
    def f(h, d):
        out = 0
        for fn in fns:
            out = out + 1.0 / (k + ranks_matrix(np.asarray(fn(h, d), dtype=np.float64)))
        return out
    return f


def cascade(rule_fn, emb_fn):
    def f(h, d):
        e = np.asarray(emb_fn(h, d), dtype=np.float64)
        e = (e - e.min(1, keepdims=True)) / (np.ptp(e, 1, keepdims=True) + 1e-9)
        return rule_fn(h, d) * 10.0 + e
    return f


def main():
    res = {}
    # ---- WN18RR ----
    kg = data.load("wn18rr")
    preds = anyburl.parse(ROOT / "runs/anyburl_wn18rr/preds-100")
    A = anyburl_scores(kg, preds)
    rot = kge.score_fn(torch.load(ROOT / "runs/wn18rr_RotatE_s0.pt", weights_only=False, map_location="mps"), kg.num_relations)
    tra = kge.score_fn(torch.load(ROOT / "runs/wn18rr_TransE_s0.pt", weights_only=False, map_location="mps"), kg.num_relations)
    methods = {"AnyBURL": A, "RotatE": rot, "RRF(AnyBURL, RotatE)": rrf(A, rot),
               "RRF(AnyBURL, RotatE, TransE)": rrf(A, rot, tra), "Cascade(AnyBURL -> RotatE)": cascade(A, rot)}
    res["WN18RR"] = {}
    for k, f in methods.items():
        res["WN18RR"][k] = evaluate.summarize(evaluate.ranks(kg, f, kg.test))
        log("WN18RR", k, res["WN18RR"][k])
    (RESULTS / "exp13_ensembles.json").write_text(json.dumps(res, indent=2))

    # ---- GO prospective ----
    from exp4_go_temporal import Reasoner, TextScorer
    kg, info, kinds, _, terms = go.temporal("2020-01-01", "2026-08-05")
    preds = anyburl.parse(ROOT / "runs/anyburl_go_2020_2026/preds-1000")
    A = anyburl_scores(kg, preds)
    tag = "2020-01-01_2026-08-05"
    rot = kge.score_fn(torch.load(ROOT / f"runs/go_{tag}_RotatE_s0.pt", weights_only=False, map_location="mps"), kg.num_relations)
    tra = kge.score_fn(torch.load(ROOT / f"runs/go_{tag}_TransE_s0.pt", weights_only=False, map_location="mps"), kg.num_relations)
    txt = TextScorer(kg, terms, model="BAAI/bge-large-en-v1.5").score  # hosted bge-large
    rea = Reasoner(kg).score
    methods = {"AnyBURL": A, "Reasoner": rea, "Text (bge-large)": txt, "RotatE": rot, "TransE": tra,
               "RRF(AnyBURL, RotatE)": rrf(A, rot),
               "RRF(RotatE, TransE, Text)": rrf(rot, tra, txt),
               "RRF(AnyBURL, RotatE, TransE, Text)": rrf(A, rot, tra, txt),
               "RRF(AnyBURL, RotatE, TransE, Text, Reasoner)": rrf(A, rot, tra, txt, rea)}
    res["GO 2020->2026"] = {}
    qk = np.concatenate([kinds, kinds])
    for k, f in methods.items():
        r = evaluate.ranks(kg, f, kg.test)
        n = len(kinds)
        res["GO 2020->2026"][k] = {"all": evaluate.summarize(r), "tail_only": evaluate.summarize(r[:n]),
                                   "novel_tail_only": evaluate.summarize(r[:n][kinds == "novel"]),
                                   "entailed_tail_only": evaluate.summarize(r[:n][kinds == "entailed"])}
        m = res["GO 2020->2026"][k]
        log("GO", f"{k:45s} tail MRR={m['tail_only']['MRR']:.3f} H1={m['tail_only']['Hits@1']:.3f} H10={m['tail_only']['Hits@10']:.3f}"
            f" | novel H10={m['novel_tail_only']['Hits@10']:.3f} | entailed H10={m['entailed_tail_only']['Hits@10']:.3f}")
        (RESULTS / "exp13_ensembles.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
