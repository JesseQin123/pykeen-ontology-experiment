"""Experiment 10 - New concepts: where does a term created after the training snapshot belong?

Standard PyKEEN models are transductive: a GO term created after 2020 has no embedding. In production this is
the common case (new concepts arrive daily). We take terms that exist in the 2026 release but not in 2020 and
whose is_a parents already existed in 2020, and rank candidate parents with:
  * text similarity (term name, name+definition; small and large sentence encoders)
  * a text->graph bridge: ridge regression from text embeddings to the 2020 RotatE embeddings, then RotatE's
    own is_a scoring (an inductive use of a transductive model, no retraining)
  * fusion of the two, and local-LLM re-ranking of the fused top-20
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import go  # noqa: E402

RESULTS, RUNS = ROOT / "results", ROOT / "runs"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def metrics(ranks):
    r = np.asarray(ranks, float)
    return {"Hits@1": float(np.mean(r <= 1)), "Hits@10": float(np.mean(r <= 10)), "MRR": float(np.mean(1 / r)), "n": len(r)}


def main(old="2020-01-01", new="2026-08-05", llm="qwen3.5:27b", llm_n=150):
    kg, _, _, _, t_old = go.temporal(old, new)
    t_new, e_new = go.parse(new)
    eid = {e: i for i, e in enumerate(kg.entity_labels)}
    new_terms = [i for i, t in t_new.items() if not t.obsolete and i not in t_old]
    queries = {}
    for h, r, t in e_new:
        if r == "is_a" and h in set(new_terms) and t in eid:
            queries.setdefault(h, set()).add(eid[t])
    qs = sorted(queries)
    log("new terms", len(new_terms), "with is_a parent in 2020 graph", len(qs))

    from sentence_transformers import SentenceTransformer
    sims, res = {}, {"n_new_terms": len(new_terms), "n_queries": len(qs), "methods": {}}
    old_txt = lambda e, d: t_old[e].name + ((": " + t_old[e].definition) if d else "")
    new_txt = lambda e, d: t_new[e].name + ((": " + t_new[e].definition) if d else "")
    enc = {}
    for model in ("BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5"):
        st = SentenceTransformer(model, device="mps")
        for d in (False, True):
            O = st.encode([old_txt(e, d) for e in kg.entity_labels], normalize_embeddings=True, batch_size=256)
            N = st.encode([new_txt(e, d) for e in qs], normalize_embeddings=True, batch_size=256)
            enc[(model, d)] = (O, N)
            sims[f"text {model.split('/')[-1]}{' +def' if d else ''}"] = N @ O.T
        log("encoded", model)

    # text -> RotatE bridge (inductive placement without retraining)
    m = torch.load(RUNS / f"go_{old}_{new}_RotatE_s0.pt", weights_only=False, map_location="cpu")
    ent = m.entity_representations[0](indices=None).detach()
    rel = m.relation_representations[0](indices=None).detach()
    Xr = torch.cat([ent.real, ent.imag], 1).numpy()
    O, N = enc[("BAAI/bge-large-en-v1.5", True)]
    lam = 1.0
    W = np.linalg.solve(O.T @ O + lam * np.eye(O.shape[1]), O.T @ Xr)
    Hn = N @ W
    d = ent.shape[1]
    h = torch.complex(torch.as_tensor(Hn[:, :d]), torch.as_tensor(Hn[:, d:]))
    r = rel[kg.relation_labels.index("is_a")]
    S = []
    for s in range(0, len(h), 64):
        hr = (h[s : s + 64] * r)[:, None, :]
        S.append(-(hr - ent[None]).abs().pow(2).sum(-1).sqrt().numpy())
    sims["bridge: text->RotatE (is_a scoring)"] = np.concatenate(S)
    z = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
    sims["fusion: bge-large +def & bridge"] = z(sims["text bge-large-en-v1.5 +def"]) + 0.5 * z(sims["bridge: text->RotatE (is_a scoring)"])

    # popularity prior on is_a parents helps text pick general-enough parents
    isa = kg.relation_labels.index("is_a")
    pop = np.bincount(kg.train[kg.train[:, 1] == isa][:, 2], minlength=kg.num_entities)
    sims["fusion + parent-popularity prior"] = sims["fusion: bge-large +def & bridge"] + 0.3 * z(np.log1p(pop)[None, :].repeat(len(qs), 0))

    def ranks_of(Sm):
        out = []
        for i, q in enumerate(qs):
            row = Sm[i].copy()
            golds = queries[q]
            for g in golds:
                others = [x for x in golds if x != g]
                rr = row.copy(); rr[others] = -np.inf
                out.append((rr > rr[g]).sum() + 1)
        return out

    for k, Sm in sims.items():
        res["methods"][k] = metrics(ranks_of(Sm))
        log(f"{k:44s}", res["methods"][k])
    (RESULTS / "exp10_new_terms.json").write_text(json.dumps(res, indent=2))

    # LLM re-rank of the best fused top-20
    best = max(res["methods"], key=lambda k: res["methods"][k]["MRR"])
    Sm = sims[best]
    rng = np.random.default_rng(0)
    pick = rng.permutation(len(qs))[:llm_n]
    base = ok = intop = 0
    ex = []
    for i in pick:
        q = qs[i]
        top = list(np.argsort(-Sm[i])[:20])
        base += top[0] in queries[q]
        intop += any(t in queries[q] for t in top)
        opts = "\n".join(f"{j+1}. {t_old[kg.entity_labels[c]].name}" for j, c in enumerate(top))
        prompt = (f"New Gene Ontology term: \"{t_new[q].name}\"\nDefinition: {t_new[q].definition}\n\n"
                  f"Which ONE of these existing GO terms is the most appropriate direct parent (is_a)?\n{opts}\n\nAnswer with the number only.")
        body = json.dumps({"model": llm, "prompt": prompt, "stream": False, "think": False,
                           "options": {"temperature": 0, "num_predict": 8}}).encode()
        try:
            a = json.loads(urllib.request.urlopen(urllib.request.Request(
                "http://localhost:11434/api/generate", data=body, headers={"Content-Type": "application/json"}), timeout=600).read())["response"]
            k = int("".join(ch for ch in a if ch.isdigit())[:2] or 0) - 1
        except Exception:  # noqa: BLE001
            k = -1
        hit = 0 <= k < 20 and top[k] in queries[q]
        ok += hit
        if len(ex) < 20:
            ex.append({"new_term": t_new[q].name, "gold": [t_old[kg.entity_labels[g]].name for g in queries[q]],
                       "fused_top1": t_old[kg.entity_labels[top[0]]].name,
                       "llm": t_old[kg.entity_labels[top[k]]].name if 0 <= k < 20 else "?", "llm_ok": bool(hit)})
    res["llm_rerank"] = {"base_method": best, "model": llm, "n": len(pick), "gold_in_top20": intop / len(pick),
                         "base_Hits@1(any parent)": base / len(pick), "llm_Hits@1(any parent)": ok / len(pick), "examples": ex}
    log("llm", {k: v for k, v in res["llm_rerank"].items() if k != "examples"})
    (RESULTS / "exp10_new_terms.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
