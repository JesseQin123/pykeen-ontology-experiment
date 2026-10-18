"""Experiment 11 - Re-run the embedding-vs-rules comparison with the real AnyBURL (the literature's strong baseline).

Our own path-rule learner (length <= 2) was a weak stand-in. AnyBURL learns cyclic and acyclic rules up to
length 3 with reinforcement-learned sampling. Same filtered realistic-rank protocol as every other method.
Datasets: WN18RR (random hold-out) and the GO 2020 -> 2026 prospective split.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import anyburl, data, evaluate, go  # noqa: E402

RESULTS = ROOT / "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    res = {}
    kg = data.load("wn18rr")
    for secs in (100, 1000):
        t = time.time()
        preds = anyburl.run(kg, ROOT / "runs" / "anyburl_wn18rr", seconds=secs)
        r = anyburl.ranks(kg, preds)
        res[f"WN18RR AnyBURL {secs}s"] = {**evaluate.summarize(r), "wall_seconds": time.time() - t}
        np.save(RESULTS / f"exp11_anyburl_wn18rr_{secs}s_ranks.npy", r)
        log(res)
        (RESULTS / "exp11_anyburl.json").write_text(json.dumps(res, indent=2))
    kg, info, kinds, _, _ = go.temporal("2020-01-01", "2026-08-05")
    t = time.time()
    preds = anyburl.run(kg, ROOT / "runs" / "anyburl_go_2020_2026", seconds=1000)
    r = anyburl.ranks(kg, preds)
    qk = np.concatenate([kinds, kinds])
    res["GO 2020->2026 AnyBURL 1000s"] = {"all": evaluate.summarize(r), "tail_only": evaluate.summarize(r[: len(kinds)]),
                                         "novel": evaluate.summarize(r[qk == "novel"]),
                                         "entailed": evaluate.summarize(r[qk == "entailed"]),
                                         "novel_tail_only": evaluate.summarize(r[: len(kinds)][kinds == "novel"]),
                                         "wall_seconds": time.time() - t}
    log(res)
    (RESULTS / "exp11_anyburl.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
