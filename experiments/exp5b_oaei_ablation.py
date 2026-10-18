"""Experiment 5b - Does the graph embedding earn its place in the final OAEI Anatomy pipeline?

Same fully automatic pipeline as exp 5 (exact-label anchors -> mutual-best candidates -> local-LLM yes/no check),
with the candidate scorer swapped:
  (a) text only (bge-large + synonyms)
  (b) text + joint RotatE structure (the best non-LLM ranker in exp 5)
  (c) text + joint RotatE + anchor-neighbourhood + lexical (exp 5's original hybrid)
Reports P/R/F1 against the OAEI reference with and without the LLM check, for each scorer.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp5_oaei_anatomy import (anchor_neigh, cos, describe, lexical_sim, ollama, text_sim,  # noqa: E402
                               train_struct, zs)
from kgv import oaei  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    M, H = oaei.load("mouse"), oaei.load("human")
    R = set(oaei.reference())
    mi = {c: i for i, c in enumerate(M.classes)}
    hi = {c: i for i, c in enumerate(H.classes)}
    lab_h: dict[str, list[str]] = {}
    for c, l in H.label.items():
        lab_h.setdefault(l, []).append(c)
    anchors = [(c, lab_h[l][0]) for c, l in M.label.items() if l in lab_h and len(lab_h[l]) == 1]
    amap = {b: a for a, b in anchors}
    Mt = [(f"M:{h}", f"M:{r}", f"M:{t}") for h, r, t in M.edges]
    Jt = Mt + [(f"M:{amap[h]}" if h in amap else f"H:{h}", f"H:{r}", f"M:{amap[t]}" if t in amap else f"H:{t}")
               for h, r, t in H.edges]
    tfJ, EJ = train_struct(Jt, 0)
    get = lambda pre, c: EJ[tfJ.entity_to_id.get(f"{pre}:{c}", tfJ.entity_to_id.get(f"M:{amap.get(c, '')}", 0))]
    S_joint = cos(np.stack([get("M", c) for c in M.classes]), np.stack([get("H", c) for c in H.classes]))
    S_text = text_sim("BAAI/bge-large-en-v1.5", M, H, with_syn=True)
    S_nb = anchor_neigh(M, H, anchors)
    S_lex = lexical_sim(M, H)
    scorers = {
        "text only (bge-large+syn)": S_text,
        "text + joint RotatE": zs(S_text) + 0.5 * zs(S_joint),
        "text + joint RotatE + anchor-neigh + lexical": zs(S_text) + 0.5 * zs(S_joint) + 0.5 * zs(S_nb) + 0.5 * zs(S_lex),
    }
    parM, parH = {}, {}
    for h, r, t in M.edges:
        parM.setdefault(h, []).append((h, r, t))
    for h, r, t in H.edges:
        parH.setdefault(h, []).append((h, r, t))

    def prf(pairs):
        P = set(pairs)
        tp = len(P & R)
        p, r = tp / max(len(P), 1), tp / len(R)
        return {"pairs": len(P), "precision": p, "recall": r, "F1": 2 * p * r / max(p + r, 1e-9)}

    anchored_h = {b for _, b in anchors}
    cand_cols = np.array([hi[c] for c in H.classes if c not in anchored_h])
    unm = [c for c in M.classes if c not in {a for a, _ in anchors}]
    um_rows = np.array([mi[c] for c in unm])
    verdict_cache: dict = {}
    res = {}
    for name, S in scorers.items():
        Sx = S[um_rows][:, cand_cols]
        col_best = Sx.argmax(0)
        cands = [(c, H.classes[cand_cols[int(Sx[i].argmax())]]) for i, c in enumerate(unm)
                 if col_best[int(Sx[i].argmax())] == i]
        kept = []
        for a, b in cands:
            if (a, b) not in verdict_cache:
                p = (f"Mouse anatomy term: {describe(M, a, parM)}\nHuman anatomy term: {describe(H, b, parH)}\n\n"
                     "Do these denote the same anatomical entity (an equivalence mapping between the two ontologies)? "
                     "Answer yes or no.")
                try:
                    verdict_cache[(a, b)] = ollama(p, n=3).strip().lower().startswith("y")
                except Exception:  # noqa: BLE001
                    verdict_cache[(a, b)] = False
            if verdict_cache[(a, b)]:
                kept.append((a, b))
        res[name] = {"mutual_best_candidates": len(cands),
                     "candidate_precision": sum(p in R for p in cands) / max(len(cands), 1),
                     "no_LLM": prf(anchors + cands), "with_LLM_check": prf(anchors + kept),
                     "llm_kept": len(kept)}
        log(name, res[name])
        (ROOT / "results" / "exp5b_oaei_ablation.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
