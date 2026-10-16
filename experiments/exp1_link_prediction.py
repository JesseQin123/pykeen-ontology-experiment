"""Experiment 1 - Do graph embeddings find real, missing relations in a semantic ontology (WordNet)?

Questions, each answered against a non-embedding baseline rather than in isolation:
  Q1  Accuracy: filtered link prediction vs. popularity and an explicit path-rule learner ("graph search").
  Q2  Beyond search: on held-out edges that NO length<=2 rule path reaches, do embeddings still recover them?
  Q3  Complementarity: does a rules-first / embedding-fallback hybrid beat either alone?
  Q4  Suggestions: the top suggestion per query that is absent from the training graph - how often is it real?
      Checked against the held-out split AND the full WordNet 3.0 database (open-world check).
  Q5  Robustness: seed variance and a paired bootstrap CI on the embedding-vs-rules gap.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import baselines, data, evaluate, kge, wordnet  # noqa: E402

DEVICE = "mps"
RUNS, RESULTS = ROOT / "runs", ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODELS = [  # (name, seed, overrides)
    ("RotatE", 0, dict(lr=1e-2, epochs=40)),
    ("RotatE", 1, dict(lr=1e-2, epochs=40)),
    ("RotatE", 2, dict(lr=1e-2, epochs=40)),
    ("TransE", 0, dict(lr=1e-3, epochs=100)),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def per_relation(kg, r):
    q = data.as_queries(kg.test, kg.num_relations)
    rel = q[:, 1] % kg.num_relations
    return {kg.relation_labels[i]: dict(evaluate.summarize(r[rel == i]), share=float(np.mean(rel == i)))
            for i in range(kg.num_relations)}


def suggestions(kg, score, verifier, train_known):
    """Top-1 candidate per test query that is not already in the training graph, plus its verdict."""
    q = data.as_queries(kg.test, kg.num_relations)
    uq = np.unique(q[:, :2], axis=0)
    heldout = kg.known_tails()
    R = kg.num_relations
    rows = []
    for s in range(0, len(uq), 256):
        b = uq[s : s + 256]
        sc = np.asarray(score(b[:, 0], b[:, 1]), dtype=np.float64)
        for i, (h, d) in enumerate(b):
            row = sc[i]
            row[list(train_known.get((int(h), int(d)), ()))] = -np.inf
            row[h] = -np.inf
            c = int(np.argmax(row))
            if c in heldout.get((int(h), int(d)), ()):
                v = "held-out"
            else:
                trip = (int(h), int(d), c) if d < R else (c, int(d) - R, int(h))
                v = verifier.verdict(*trip)
            rows.append((float(row[c]), v, int(h), int(d), c))
    rows.sort(key=lambda x: -x[0])
    out = {}
    for frac in (0.1, 0.25, 0.5, 1.0):
        top = rows[: max(1, int(len(rows) * frac))]
        cnt = Counter(v for _, v, *_ in top)
        n = len(top)
        out[f"top{int(frac*100)}%"] = {
            "n": n,
            "held_out": cnt["held-out"] / n,
            "true_in_wordnet_direct": cnt["direct"] / n,
            "entailed_by_wordnet": cnt["entailed"] / n,
            "unverified": cnt["unverified"] / n,
            "precision_lower_bound": (cnt["held-out"] + cnt["direct"] + cnt["entailed"]) / n,
        }
    return out, rows


def paired_bootstrap(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    diff = 1 / a - 1 / b
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    m = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    kg = data.load("wn18rr")
    log("loaded", kg.num_entities, "entities", len(kg.train), "train triples")
    verifier = wordnet.Verifier(kg)
    train_known: dict = {}
    R = kg.num_relations
    for h, r, t in kg.train:
        train_known.setdefault((int(h), int(r)), set()).add(int(t))
        train_known.setdefault((int(t), int(r) + R), set()).add(int(h))

    res = {"dataset": "WN18RR", "verifier_agreement_on_train": verifier.agreement, "methods": {}}
    ranks = {}

    pop = baselines.Popularity(kg)
    ranks["Popularity"] = evaluate.ranks(kg, pop.score, kg.test)
    t = time.time()
    rules = baselines.PathRules(kg)
    res["rules_build_seconds"] = time.time() - t
    res["top_rules"] = rules.describe(4)
    ranks["PathRules"] = evaluate.ranks(kg, rules.score, kg.test)
    log("baselines done", evaluate.summarize(ranks["PathRules"]))

    models = {}
    for name, seed, ov in MODELS:
        ov = dict(ov)
        epochs = ov.pop("epochs")
        m, info = kge.train(kg, name, seed=seed, device=DEVICE, epochs=epochs,
                            cache=RUNS / f"wn18rr_{name}_s{seed}.pt", **ov)
        key = f"{name}_s{seed}"
        models[key] = m
        ranks[key] = evaluate.ranks(kg, kge.score_fn(m, R), kg.test)
        res.setdefault("train_info", {})[key] = info
        log(key, info, evaluate.summarize(ranks[key]))

    # Q3 hybrid: rule-supported candidates first (ordered by rule confidence), embedding orders everything else
    best = "RotatE_s0"
    kfn = kge.score_fn(models[best], R)

    def hybrid(h, d):
        k = kfn(h, d).astype(np.float64)
        k = (k - k.min(1, keepdims=True)) / (np.ptp(k, 1, keepdims=True) + 1e-9)
        return rules.score(h, d) * 1e3 + k

    ranks["Hybrid(rules→RotatE)"] = evaluate.ranks(kg, hybrid, kg.test)

    for k, r in ranks.items():
        res["methods"][k] = evaluate.summarize(r)
    rot = [k for k in ranks if k.startswith("RotatE")]
    res["RotatE_seed_mean_std"] = {
        m: [float(np.mean([res["methods"][k][m] for k in rot])), float(np.std([res["methods"][k][m] for k in rot]))]
        for m in ("MRR", "Hits@1", "Hits@10")
    }
    res["bootstrap_MRR_diff_RotatE_minus_rules"] = dict(zip(("mean", "ci95_lo", "ci95_hi"),
                                                            paired_bootstrap(ranks[best], ranks["PathRules"])))
    res["per_relation"] = {k: per_relation(kg, ranks[k]) for k in ("PathRules", best, "TransE_s0")}

    # Q2 reachability: could explicit search (any learned rule path) have found the held-out edge at all?
    q = data.as_queries(kg.test, R)
    reach = rules.reachable(q[:, 0], q[:, 1], q[:, 2])
    res["reachability"] = {"share_reachable_by_rules": float(reach.mean())}
    for k in ("PathRules", best, "TransE_s0", "Popularity"):
        res["reachability"][k] = {"reachable": evaluate.summarize(ranks[k][reach]),
                                  "unreachable": evaluate.summarize(ranks[k][~reach])}
    hk, hr = ranks[best] <= 10, ranks["PathRules"] <= 10
    res["complementarity_hits10"] = {"both": float(np.mean(hk & hr)), "embedding_only": float(np.mean(hk & ~hr)),
                                     "rules_only": float(np.mean(~hk & hr)), "neither": float(np.mean(~hk & ~hr))}

    # Q4 suggestion quality, open-world verified against WordNet 3.0
    res["suggestions"] = {}
    examples = {}
    for name, fn in (("PathRules", rules.score), (best, kfn), ("Popularity", pop.score)):
        summ, rows = suggestions(kg, fn, verifier, train_known)
        res["suggestions"][name] = summ
        ex = [r for r in rows if r[1] in ("direct", "entailed", "unverified")][:400]
        examples[name] = [
            {"verdict": v, "head": verifier.label(h), "relation": kg.relation_labels[d % R] + ("^-1" if d >= R else ""),
             "suggested": verifier.label(c)} for _, v, h, d, c in ex[::10]
        ]
        log("suggestions", name, summ["top25%"])
    res["suggestion_examples"] = examples

    np.savez_compressed(RESULTS / "exp1_ranks.npz", **{k.replace("→", "_"): v for k, v in ranks.items()})
    (RESULTS / "exp1_wn18rr.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
