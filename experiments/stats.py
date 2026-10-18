"""Paired bootstrap 95% CIs for the key head-to-head comparisons (MRR and Hits@10 differences)."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results"


def boot(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, f in (("MRR", lambda r: 1 / r), ("Hits@10", lambda r: (r <= 10).astype(float))):
        d = f(a) - f(b)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        m = d[idx].mean(1)
        out[name] = {"diff": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
    out["n"] = int(len(a))
    return out


res = {}
wn = np.load(R / "exp1_ranks.npz")
ab = np.load(R / "exp11_anyburl_wn18rr_100s_ranks.npy")
res["WN18RR: AnyBURL - RotatE"] = boot(ab, wn["RotatE_s0"])
res["WN18RR: RotatE - own PathRules"] = boot(wn["RotatE_s0"], wn["PathRules"])
g = np.load(R / "exp4_ranks_2020-01-01_2026-08-05.npz")
k = g["kinds"]
n = len(k)
tail = lambda x: x[:n]
for a, b in (("Hybrid(RotatE+Text)", "RotatE"), ("Hybrid(RotatE+Text)", "Text(bge-small name)"),
             ("RotatE", "PathRules"), ("TransE", "RotatE"), ("Reasoner(is_a transitivity)", "Hybrid(RotatE+Text)")):
    res[f"GO 2020->2026 tail: {a} - {b}"] = boot(tail(g[a]), tail(g[b]))
    res[f"GO 2020->2026 novel tail: {a} - {b}"] = boot(tail(g[a])[k == "novel"], tail(g[b])[k == "novel"])
(R / "STATS.json").write_text(json.dumps(res, indent=1))
for key, v in res.items():
    print(f"{key:70s} dMRR={v['MRR']['diff']:+.3f} [{v['MRR']['ci95'][0]:+.3f},{v['MRR']['ci95'][1]:+.3f}]  dH10={v['Hits@10']['diff']:+.3f} [{v['Hits@10']['ci95'][0]:+.3f},{v['Hits@10']['ci95'][1]:+.3f}]")
