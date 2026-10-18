"""Experiment 14 - Where do LLMs stand, and does structure still add value on top of them?

Subcommands
  dump     write the exact re-ranking question sets used in exp 4 / 5 / 10 (candidates without gold) to
           research/llm_questions/*.json, and the gold answers to runs/llm_gold/*.json (kept apart on purpose)
  gen      LLM-only baseline, no candidates: the local LLM names the answer, which is mapped to the nearest
           ontology term by a sentence encoder (Hits@1 / Hits@5 over 5 generated names)
  probe    contamination probe: can the LLM recall GO IDs of terms created after its plausible cutoff vs older terms?
  score    score an answers file {"id": choice_number} produced by any judge (e.g. Claude sub-agents)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
QDIR, GDIR = ROOT / "research" / "llm_questions", ROOT / "runs" / "llm_gold"
RESULTS = ROOT / "results"
LLM = "qwen3.5:27b"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ollama(prompt, n=64):
    body = json.dumps({"model": LLM, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": n}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["response"]


# ---------------------------------------------------------------- question sets
def go_novel_isa(n=150):
    """Exp 4's LLM set: 150 novel is_a additions (2020->2026), top-20 from Hybrid(RotatE+Text)."""
    from exp4_go_temporal import TextScorer, zfuse
    from kgv import go, kge
    kg, _, kinds, _, terms = go.temporal("2020-01-01", "2026-08-05")
    tag = "2020-01-01_2026-08-05"
    rot = kge.score_fn(torch.load(ROOT / f"runs/go_{tag}_RotatE_s0.pt", weights_only=False, map_location="mps"), kg.num_relations)
    fused = zfuse(rot, TextScorer(kg, terms).score)
    isa = kg.relation_labels.index("is_a")
    idx = [i for i in range(len(kg.test)) if kinds[i] == "novel" and kg.test[i, 1] == isa]
    idx = np.random.default_rng(0).permutation(idx)[:n]
    known = kg.known_tails()
    qs, gold = [], {}
    for i in idx:
        h, r, t = map(int, kg.test[i])
        sc = fused(np.array([h]), np.array([r]))[0]
        sc[[x for x in known[(h, r)] if x != t]] = -np.inf
        sc[h] = -np.inf
        top = [int(c) for c in np.argsort(-sc)[:20]]
        th = terms[kg.entity_labels[h]]
        qid = f"go_novel_{i}"
        qs.append({"id": qid, "term": th.name, "definition": th.definition,
                   "candidates": [terms[kg.entity_labels[c]].name for c in top]})
        gold[qid] = {"gold_index": top.index(t) + 1 if t in top else 0, "gold_name": terms[kg.entity_labels[t]].name}
    return "Gene Ontology term (existing in 2020). Pick the ONE candidate that is its most appropriate direct parent (is_a).", qs, gold


def go_new_terms(n=150):
    """Exp 10's LLM set: 150 terms created after 2020, top-20 from the fused text+bridge+prior ranker."""
    import exp10_new_terms as e10
    from kgv import go
    kg, _, _, _, t_old = go.temporal("2020-01-01", "2026-08-05")
    t_new, e_new = go.parse("2026-08-05")
    eid = {e: i for i, e in enumerate(kg.entity_labels)}
    new_terms = {i for i, t in t_new.items() if not t.obsolete and i not in t_old}
    queries = {}
    for h, r, t in e_new:
        if r == "is_a" and h in new_terms and t in eid:
            queries.setdefault(h, set()).add(eid[t])
    qs_ids = sorted(queries)
    from kgv import textenc
    O = textenc.encode([t_old[e].name + ": " + t_old[e].definition for e in kg.entity_labels])
    N = textenc.encode([t_new[e].name + ": " + t_new[e].definition for e in qs_ids])
    m = torch.load(ROOT / "runs/go_2020-01-01_2026-08-05_RotatE_s0.pt", weights_only=False, map_location="cpu")
    ent = m.entity_representations[0](indices=None).detach()
    rel = m.relation_representations[0](indices=None).detach()
    Xr = torch.cat([ent.real, ent.imag], 1).numpy()
    W = np.linalg.solve(O.T @ O + np.eye(O.shape[1]), O.T @ Xr)
    Hn = N @ W
    d = ent.shape[1]
    h = torch.complex(torch.as_tensor(Hn[:, :d]), torch.as_tensor(Hn[:, d:]))
    r = rel[kg.relation_labels.index("is_a")]
    B = np.concatenate([-(((h[s:s + 64] * r)[:, None, :] - ent[None]).abs().pow(2).sum(-1).sqrt()).numpy() for s in range(0, len(h), 64)])
    z = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
    isa = kg.relation_labels.index("is_a")
    pop = np.bincount(kg.train[kg.train[:, 1] == isa][:, 2], minlength=kg.num_entities)
    S = z(N @ O.T) + 0.5 * z(B) + 0.3 * z(np.log1p(pop)[None, :].repeat(len(qs_ids), 0))
    pick = np.random.default_rng(0).permutation(len(qs_ids))[:n]
    qs, gold = [], {}
    for i in pick:
        q = qs_ids[i]
        top = [int(c) for c in np.argsort(-S[i])[:20]]
        qid = f"go_new_{q.replace(':', '_')}"
        qs.append({"id": qid, "term": t_new[q].name, "definition": t_new[q].definition,
                   "candidates": [t_old[kg.entity_labels[c]].name for c in top]})
        gold[qid] = {"gold_indices": [top.index(g) + 1 for g in queries[q] if g in top],
                     "gold_names": [t_old[kg.entity_labels[g]].name for g in queries[q]]}
    return "NEW Gene Ontology term (created after 2020). Pick the ONE candidate that is its most appropriate direct parent (is_a).", qs, gold


def oaei_nontrivial():
    """Exp 5's LLM set: 580 non-trivial mouse->human pairs, top-10 from the 4-way hybrid."""
    import exp5_oaei_anatomy as e5
    from kgv import oaei
    M, H = oaei.load("mouse"), oaei.load("human")
    ref = oaei.reference()
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
    SJ = e5.cos(np.stack([get("M", c) for c in M.classes]), np.stack([get("H", c) for c in H.classes]))
    S = (e5.zs(e5.text_sim("BAAI/bge-large-en-v1.5", M, H, True)) + 0.5 * e5.zs(SJ)
         + 0.5 * e5.zs(e5.anchor_neigh(M, H, anchors)) + 0.5 * e5.zs(e5.lexical_sim(M, H)))
    anchored_h = {b for _, b in anchors}
    cand_cols = np.array([hi[c] for c in H.classes if c not in anchored_h])
    col_pos = {c: j for j, c in enumerate(cand_cols)}
    pairs = [(a, b) for a, b in ref if M.label[a] != H.label[b] and hi[b] in col_pos]
    parM, parH = {}, {}
    for h, r, t in M.edges:
        parM.setdefault(h, []).append((h, r, t))
    for h, r, t in H.edges:
        parH.setdefault(h, []).append((h, r, t))
    qs, gold = [], {}
    for a, b in pairs:
        row = S[mi[a]][cand_cols]
        top = [int(c) for c in np.argsort(-row)[:10]]
        g = col_pos[hi[b]]
        qid = f"oaei_{a.split('#')[-1]}"
        qs.append({"id": qid, "term": e5.describe(M, a, parM),
                   "candidates": [e5.describe(H, H.classes[cand_cols[c]], parH) for c in top]})
        gold[qid] = {"gold_index": top.index(g) + 1 if g in top else 0, "gold_name": H.label[b]}
    return ("Mouse anatomy term (Adult Mouse Anatomy). Pick the ONE human anatomy term (NCI Thesaurus) that denotes the "
            "SAME anatomical entity; answer 0 if none does."), qs, gold


SETS = {"go_novel_isa": go_novel_isa, "go_new_terms": go_new_terms, "oaei_nontrivial": oaei_nontrivial}


def dump(full=False):
    """Write question sets. full=True writes *_full sets with every query instead of the 150-query samples."""
    QDIR.mkdir(parents=True, exist_ok=True); GDIR.mkdir(parents=True, exist_ok=True)
    for base, fn in SETS.items():
        if full and base == "oaei_nontrivial":
            continue  # already complete (all 580 non-trivial pairs)
        name = base + ("_full" if full else "")
        instr, qs, gold = fn(10**9) if full else fn()
        (QDIR / f"{name}.json").write_text(json.dumps({"instruction": instr, "questions": qs}, indent=1))
        (GDIR / f"{name}.json").write_text(json.dumps(gold, indent=1))
        log(name, len(qs), "upper bound (gold in candidates):",
            np.mean([bool(g.get("gold_index") or g.get("gold_indices")) for g in gold.values()]))


def correct(g, choice):
    if "gold_indices" in g:
        return choice in g["gold_indices"]
    return g["gold_index"] > 0 and choice == g["gold_index"]


def score(answers_path, name):
    ans = json.loads(Path(answers_path).read_text())
    gold = json.loads((GDIR / f"{name}.json").read_text())
    hits = [correct(gold[q], int(a)) for q, a in ans.items() if q in gold]
    return {"answered": len(hits), "of": len(gold), "Hits@1": float(np.mean(hits)) if hits else None}


def qwen_rerank(name):
    data = json.loads((QDIR / f"{name}.json").read_text())
    out = {}
    for q in data["questions"]:
        opts = "\n".join(f"{i+1}. {c}" for i, c in enumerate(q["candidates"]))
        p = f"{data['instruction']}\n\nTerm: {q['term']}\n" + (f"Definition: {q['definition']}\n" if q.get("definition") else "") \
            + f"\nCandidates:\n{opts}\n\nAnswer with the number only."
        try:
            a = ollama(p, 8)
            out[q["id"]] = int("".join(ch for ch in a if ch.isdigit())[:2] or 0)
        except Exception:  # noqa: BLE001
            out[q["id"]] = -1
    (ROOT / "runs" / f"answers_qwen_{name}.json").write_text(json.dumps(out))
    return score(ROOT / "runs" / f"answers_qwen_{name}.json", name)


def gen():
    """LLM-only: generate 5 names, map each to the nearest ontology term; no candidate list from any graph model."""
    from sentence_transformers import SentenceTransformer
    from kgv import go, oaei
    st = SentenceTransformer("BAAI/bge-large-en-v1.5", device="mps")
    res = {}
    # GO (both sets): candidates = all 2020 GO term names
    kg, _, _, _, t_old = go.temporal("2020-01-01", "2026-08-05")
    names = [t_old[e].name for e in kg.entity_labels]
    E = st.encode(names, normalize_embeddings=True, batch_size=256)
    for name in ("go_novel_isa", "go_new_terms"):
        data = json.loads((QDIR / f"{name}.json").read_text())
        gold = json.loads((GDIR / f"{name}.json").read_text())
        h1 = h5 = 0
        for q in data["questions"]:
            p = (f"Gene Ontology term: \"{q['term']}\"\nDefinition: {q['definition']}\n\nName its most likely direct "
                 "parent term(s) in the Gene Ontology (is_a). Give up to 5 exact GO term names, one per line, best first, no numbering.")
            lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip().strip('"') for l in ollama(p, 120).splitlines() if l.strip()][:5]
            if not lines:
                continue
            V = st.encode(lines, normalize_embeddings=True)
            picks = [names[int(np.argmax(E @ v))] for v in V]
            g = set(gold[q["id"]].get("gold_names") or [gold[q["id"]]["gold_name"]])
            h1 += picks[0] in g
            h5 += any(p in g for p in picks)
        res[name] = {"n": len(data["questions"]), "LLM-only Hits@1": h1 / len(data["questions"]),
                     "LLM-only Hits@5": h5 / len(data["questions"])}
        log(name, res[name])
    # OAEI: candidates = all human classes (labels)
    H = oaei.load("human")
    hnames = [H.label[c] for c in H.classes]
    EH = st.encode(hnames, normalize_embeddings=True, batch_size=256)
    data = json.loads((QDIR / "oaei_nontrivial.json").read_text())
    gold = json.loads((GDIR / "oaei_nontrivial.json").read_text())
    h1 = h5 = 0
    for q in data["questions"]:
        p = (f"Mouse anatomy term: {q['term']}\n\nWhat is the equivalent term in the NCI Thesaurus human anatomy "
             "ontology? Give up to 5 candidate NCI Thesaurus labels, one per line, best first, no numbering.")
        lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip().strip('"').replace("_", " ").lower() for l in ollama(p, 120).splitlines() if l.strip()][:5]
        if not lines:
            continue
        V = st.encode(lines, normalize_embeddings=True)
        picks = [hnames[int(np.argmax(EH @ v))] for v in V]
        g = gold[q["id"]]["gold_name"]
        h1 += picks[0] == g
        h5 += g in picks
    res["oaei_nontrivial"] = {"n": len(data["questions"]), "LLM-only Hits@1": h1 / len(data["questions"]),
                              "LLM-only Hits@5": h5 / len(data["questions"])}
    log("oaei", res["oaei_nontrivial"])
    return res


def probe(n=100):
    """Ask for the GO ID of a term name. Terms created 2024-2026 vs terms that existed in 2018."""
    from kgv import go
    t18, _ = go.parse("2018-01-01")
    t24, _ = go.parse("2024-01-17")
    t26, _ = go.parse("2026-08-05")
    rng = np.random.default_rng(0)
    recent = [i for i, t in t26.items() if not t.obsolete and i not in t24]
    old = [i for i, t in t18.items() if not t.obsolete and i in t26 and not t26[i].obsolete]
    res = {}
    for label, pool in (("created 2024-2026", recent), ("existed in 2018", old)):
        sel = rng.choice(pool, min(n, len(pool)), replace=False)
        ok = 0
        for i in sel:
            a = ollama(f"What is the Gene Ontology identifier (GO:XXXXXXX) of the term \"{t26[i].name}\"? Answer with the ID only.", 16)
            m = re.search(r"GO:\d{7}", a)
            ok += bool(m and m.group(0) == i)
        res[label] = {"n": len(sel), "exact_ID_recall": ok / len(sel)}
        log("probe", label, res[label])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "dump_full", "gen", "probe", "score", "qwen", "all_local"])
    ap.add_argument("--answers"); ap.add_argument("--set")
    a = ap.parse_args()
    out_p = RESULTS / "exp14_llm.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    if a.cmd == "dump":
        dump()
    elif a.cmd == "dump_full":
        dump(full=True)
    elif a.cmd == "score":
        res.setdefault("judges", {})[Path(a.answers).stem] = score(a.answers, a.set)
        log(res["judges"][Path(a.answers).stem])
    elif a.cmd == "all_local":
        dump()
        res["qwen_rerank_on_dumped_sets"] = {n: qwen_rerank(n) for n in SETS}
        log(res["qwen_rerank_on_dumped_sets"]); out_p.write_text(json.dumps(res, indent=2))
        res["llm_only_generation"] = gen(); out_p.write_text(json.dumps(res, indent=2))
        res["contamination_probe"] = probe()
    out_p.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
