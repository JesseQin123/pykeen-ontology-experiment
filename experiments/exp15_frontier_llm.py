"""Experiment 15 - Frontier LLMs (Claude Opus 5.5, GPT-6.1) alongside the local qwen3.5-27B.

Nothing earlier is overwritten: results go to results/exp15_frontier_llm.json.
  rerank   same question sets as exp 14 (exp 4 / 5 / 10 candidates), judged by each frontier model
  gen      LLM-only (no candidates from any graph model): name the answer, map to the ontology by embedding
  probe    contamination probe: recall of GO IDs for recently created vs old terms
  oaei     full OAEI Anatomy pipeline with the frontier model as yes/no verifier, for three candidate scorers
           (text only / text + joint RotatE / 4-way hybrid) -> does the graph embedding still add value?
The only text encoder is bge (local), as in every other experiment.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from kgv import remote, textenc  # noqa: E402

QDIR, GDIR = ROOT / "research" / "llm_questions", ROOT / "runs" / "llm_gold"
OUT = ROOT / "results" / "exp15_frontier_llm.json"
MODELS = ["claude-opus-5-5", "gpt-6.1-sol"]
EMB_MAP = "BAAI/bge-large-en-v1.5"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(res):
    OUT.write_text(json.dumps(res, indent=2))


def is_yes(s):
    return re.sub(r"^[\W_]+", "", s or "").lower().startswith("y")


def first_int(s, default=-1):
    m = re.search(r"\d+", s or "")
    return int(m.group(0)) if m else default


SETS = ["go_novel_isa", "go_new_terms", "oaei_nontrivial"]


def rerank(res):
    from exp14_llm import correct
    for name in SETS:
        p = QDIR / f"{name}.json"
        if not p.exists():
            log("missing question set", name); continue
        data = json.loads(p.read_text())
        gold = json.loads((GDIR / f"{name}.json").read_text())
        prompts = []
        for q in data["questions"]:
            opts = "\n".join(f"{i+1}. {c}" for i, c in enumerate(q["candidates"]))
            prompts.append(f"{data['instruction']}\n\nTerm: {q['term']}\n" + (f"Definition: {q['definition']}\n" if q.get("definition") else "")
                           + f"\nCandidates:\n{opts}\n\nAnswer with the number only.")
        upper = float(np.mean([bool(g.get("gold_index") or g.get("gold_indices")) for g in gold.values()]))
        for m in MODELS:
            outs = remote.chat_many(m, prompts, max_tokens=8)
            ans = {q["id"]: first_int(o) for q, o in zip(data["questions"], outs)}
            (ROOT / "runs" / f"answers_{m}_{name}.json").write_text(json.dumps(ans))
            hits = [correct(gold[k], v) for k, v in ans.items()]
            res.setdefault("rerank", {}).setdefault(name, {"n": len(hits), "gold_in_candidates": upper})[m] = float(np.mean(hits))
            log("rerank", name, m, res["rerank"][name][m]); save(res)


def gen(res):
    from kgv import go, oaei
    kg, _, _, _, t_old = go.temporal("2020-01-01", "2026-08-05")
    go_names = [t_old[e].name for e in kg.entity_labels]
    E_go = textenc.encode(go_names, EMB_MAP)
    H = oaei.load("human")
    h_names = [H.label[c] for c in H.classes]
    E_h = textenc.encode(h_names, EMB_MAP)
    clean = lambda l: "" if l.rstrip().endswith(":") else re.sub(r"[*`]", "", re.sub(r"^[\-\*\d\.\)\s]+", "", l)).strip().strip('"').replace("_", " ")
    for name in ("go_novel_isa", "go_new_terms", "oaei_nontrivial"):
        p = QDIR / f"{name}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        gold = json.loads((GDIR / f"{name}.json").read_text())
        if name.startswith("go"):
            prompts = [(f"Gene Ontology term: \"{q['term']}\"\nDefinition: {q['definition']}\n\nName its most likely direct parent "
                        "term(s) in the Gene Ontology (is_a). Give up to 5 exact GO term names, one per line, best first, no numbering, no IDs.")
                       for q in data["questions"]]
            names, E = go_names, E_go
        else:
            prompts = [(f"Mouse anatomy term: {q['term']}\n\nWhat is the equivalent term in the NCI Thesaurus human anatomy ontology? "
                        "Give up to 5 candidate NCI Thesaurus labels, one per line, best first, no numbering.") for q in data["questions"]]
            names, E = h_names, E_h
        for m in MODELS:
            outs = remote.chat_many(m, prompts, max_tokens=160)
            h1 = h5 = 0
            for q, o in zip(data["questions"], outs):
                lines = [clean(l) for l in o.splitlines() if clean(l)][:5]
                if not lines:
                    continue
                V = textenc.encode(lines, EMB_MAP)
                picks = [names[int(np.argmax(E @ v))] for v in V]
                g = gold[q["id"]]
                gs = set(g.get("gold_names") or [g["gold_name"]])
                h1 += picks[0] in gs
                h5 += any(x in gs for x in picks)
            n = len(data["questions"])
            res.setdefault("llm_only_generation", {}).setdefault(name, {"n": n})[m] = {"Hits@1": h1 / n, "Hits@5": h5 / n}
            log("gen", name, m, res["llm_only_generation"][name][m]); save(res)


def probe(res, n=100):
    from kgv import go
    t18, _ = go.parse("2018-01-01")
    t24, _ = go.parse("2024-01-17")
    t26, _ = go.parse("2026-08-05")
    rng = np.random.default_rng(0)
    pools = {"created 2024-2026": [i for i, t in t26.items() if not t.obsolete and i not in t24],
             "existed in 2018": [i for i, t in t18.items() if not t.obsolete and i in t26 and not t26[i].obsolete]}
    for label, pool in pools.items():
        sel = list(rng.choice(pool, min(n, len(pool)), replace=False))
        prompts = [f"What is the Gene Ontology identifier (GO:XXXXXXX) of the term \"{t26[i].name}\"? Answer with the ID only." for i in sel]
        for m in MODELS:
            outs = remote.chat_many(m, prompts, max_tokens=16)
            ok = sum(bool(re.search(r"GO:\d{7}", o)) and re.search(r"GO:\d{7}", o).group(0) == i for o, i in zip(outs, sel))
            res.setdefault("contamination_probe", {}).setdefault(label, {"n": len(sel)})[m] = ok / len(sel)
            log("probe", label, m, ok / len(sel)); save(res)


def oaei_pipeline(res):
    """exp 5b with a frontier verifier; same candidate scorers plus OpenAI-embedding variants."""
    import exp5_oaei_anatomy as e5
    from kgv import oaei
    M, H = oaei.load("mouse"), oaei.load("human")
    R = set(oaei.reference())
    mi = {c: i for i, c in enumerate(M.classes)}
    hi = {c: i for i, c in enumerate(H.classes)}
    lab_h = {}
    for c, l in H.label.items():
        lab_h.setdefault(l, []).append(c)
    anchors = [(c, lab_h[l][0]) for c, l in M.label.items() if l in lab_h and len(lab_h[l]) == 1]
    amap = {b: a for a, b in anchors}
    Mt = [(f"M:{h}", f"M:{r}", f"M:{t}") for h, r, t in M.edges]
    Jt = Mt + [(f"M:{amap[h]}" if h in amap else f"H:{h}", f"H:{r}", f"M:{amap[t]}" if t in amap else f"H:{t}") for h, r, t in H.edges]
    tfJ, EJ = e5.train_struct(Jt, 0)
    get = lambda pre, c: EJ[tfJ.entity_to_id.get(f"{pre}:{c}", tfJ.entity_to_id.get(f"M:{amap.get(c, '')}", 0))]
    S_joint = e5.cos(np.stack([get("M", c) for c in M.classes]), np.stack([get("H", c) for c in H.classes]))
    S_bge = e5.text_sim("BAAI/bge-large-en-v1.5", M, H, with_syn=True)
    S_nb = e5.anchor_neigh(M, H, anchors)
    S_lex = e5.lexical_sim(M, H)

    # ranking on non-trivial pairs, same protocol as exp 5
    anchored_h = {b for _, b in anchors}
    cand_cols = np.array([hi[c] for c in H.classes if c not in anchored_h])
    col_pos = {c: j for j, c in enumerate(cand_cols)}
    pairs = [(a, b) for a, b in sorted(R) if M.label[a] != H.label[b] and hi[b] in col_pos]
    rows = np.array([mi[a] for a, _ in pairs]); gold = np.array([col_pos[hi[b]] for _, b in pairs])
    rank = lambda S: e5.rank_metrics(S[rows][:, cand_cols], rows, cand_cols, gold)
    scorers = {
        "text bge-large (+syn)": S_bge,
        "text bge-large + joint RotatE": e5.zs(S_bge) + 0.5 * e5.zs(S_joint),
        "4-way hybrid (bge-large)": e5.zs(S_bge) + 0.5 * e5.zs(S_joint) + 0.5 * e5.zs(S_nb) + 0.5 * e5.zs(S_lex),
    }
    for k, S in scorers.items():
        res.setdefault("oaei_nontrivial_ranking", {})[k] = rank(S)
        log("oaei rank", k, res["oaei_nontrivial_ranking"][k])
    save(res)

    # full pipeline with frontier verification
    parM, parH = {}, {}
    for h, r, t in M.edges:
        parM.setdefault(h, []).append((h, r, t))
    for h, r, t in H.edges:
        parH.setdefault(h, []).append((h, r, t))
    unm = [c for c in M.classes if c not in {a for a, _ in anchors}]
    um_rows = np.array([mi[c] for c in unm])

    def prf(ps):
        P = set(ps); tp = len(P & R)
        p, r = tp / max(len(P), 1), tp / len(R)
        return {"pairs": len(P), "precision": p, "recall": r, "F1": 2 * p * r / max(p + r, 1e-9),
                "recall_nontrivial": sum(1 for x in P & R if M.label[x[0]] != H.label[x[1]]) /
                                      sum(1 for x in R if M.label[x[0]] != H.label[x[1]])}
    pipe_scorers = scorers
    verdicts: dict = {}
    for m in MODELS:
        for sname, S in pipe_scorers.items():
            Sx = S[um_rows][:, cand_cols]
            col_best = Sx.argmax(0)
            cands = [(c, H.classes[cand_cols[int(Sx[i].argmax())]]) for i, c in enumerate(unm) if col_best[int(Sx[i].argmax())] == i]
            todo = [p for p in cands if (m, p) not in verdicts]
            prompts = [(f"Mouse anatomy term: {e5.describe(M, a, parM)}\nHuman anatomy term: {e5.describe(H, b, parH)}\n\n"
                        "Do these denote the same anatomical entity (an equivalence mapping between the two ontologies)? "
                        "Answer yes or no.") for a, b in todo]
            for p, o in zip(todo, remote.chat_many(m, prompts, max_tokens=4)):
                verdicts[(m, p)] = is_yes(o)
            kept = [p for p in cands if verdicts[(m, p)]]
            res.setdefault("oaei_pipeline_F1", {}).setdefault(sname, {"candidates": len(cands), "no_LLM": prf(anchors + cands)})[m] = prf(anchors + kept)
            log("oaei F1", m, sname, res["oaei_pipeline_F1"][sname][m]); save(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=["oaei", "probe", "rerank", "gen"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sets", default=",".join(SETS))
    a = ap.parse_args()
    MODELS[:] = a.models.split(",")
    SETS[:] = a.sets.split(",")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for p in a.parts:
        {"rerank": rerank, "gen": gen, "probe": probe, "oaei": oaei_pipeline}[p](res)
        save(res)
    log("done")


if __name__ == "__main__":
    main()
