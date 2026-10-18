"""Experiment 5 - Real cross-ontology mapping: OAEI Anatomy (mouse MA <-> human NCI), different naming conventions.

Fully automatic pipeline, no hand-made seeds:
  1. anchors = exact label matches (~99% precise, ~61% of the reference)
  2. the hard part = the remaining reference pairs whose labels differ ("non-trivial", ~586)
  3. compare on the non-trivial pairs: structure-only (no anchors / Procrustes / joint / anchor-neighbourhood),
     lexical, sentence embeddings (several sizes, with synonyms), hybrids, and local-LLM re-ranking
  4. build a full alignment and score precision / recall / F1 against the OAEI reference
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from pykeen.triples import TriplesFactory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import kge, oaei  # noqa: E402

RESULTS = ROOT / "results"
LLM = "qwen3.5:27b"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rank_metrics(sim, rows, cols, gold_col):
    """sim: (len(rows), len(cols)); gold_col[i] = column index of the true match for row i."""
    true = sim[np.arange(len(rows)), gold_col]
    r = (sim > true[:, None]).sum(1) + 1 + ((sim == true[:, None]).sum(1) - 1) / 2
    return {"Hits@1": float(np.mean(r <= 1)), "Hits@10": float(np.mean(r <= 10)), "MRR": float(np.mean(1 / r)), "n": len(r)}


def zs(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)


def text_sim(model, M, H, with_syn=True):
    from kgv import textenc
    if model in textenc.HOSTED:
        enc = lambda xs: textenc.encode(xs, model)
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model, device="mps")
        enc = lambda xs: st.encode(xs, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    def variants(o):
        names, owner = [], []
        for i, c in enumerate(o.classes):
            for n in [o.label[c]] + (o.synonyms[c] if with_syn else []):
                names.append(n); owner.append(i)
        return names, np.array(owner)
    nm, om = variants(M)
    nh, oh = variants(H)
    em, eh = enc(nm), enc(nh)
    s = em @ eh.T
    # max over name variants of each class
    out = np.full((len(M.classes), len(H.classes)), -1.0, dtype=np.float32)
    for i in range(len(M.classes)):
        rows = s[om == i]
        np.maximum.at(out[i], oh, rows.max(0))
    return out


def lexical_sim(M, H):
    from sklearn.feature_extraction.text import TfidfVectorizer
    a = [" | ".join([M.label[c]] + M.synonyms[c]) for c in M.classes]
    b = [" | ".join([H.label[c]] + H.synonyms[c]) for c in H.classes]
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True).fit(a + b)
    return (v.transform(a) @ v.transform(b).T).toarray().astype(np.float32)


def train_struct(triples, seed, epochs=200):
    tf = TriplesFactory.from_labeled_triples(np.array(triples, dtype=str))
    m, _ = kge.train(None, "RotatE", tf=tf, seed=seed, device="mps", epochs=epochs, lr=1e-2, batch_size=256)
    return tf, kge.entity_embeddings(m)


def cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def anchor_neigh(M, H, anchors):
    """Bag of anchors within 2 hops (1 hop weighted x2), cosine. No embeddings."""
    def feats(o, idx_of_anchor):
        ix = {c: i for i, c in enumerate(o.classes)}
        rows = [ix[h] for h, _, t in o.edges] + [ix[t] for h, _, t in o.edges]
        cols = [ix[t] for h, _, t in o.edges] + [ix[h] for h, _, t in o.edges]
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ix), len(ix)))
        A = (A > 0).astype(np.float32)
        F = (2 * A + ((A @ A) > 0)).tocsr()
        sel = [ix[c] for c in idx_of_anchor]
        return F[:, sel]
    FM = feats(M, [a for a, _ in anchors])
    FH = feats(H, [b for _, b in anchors])
    n = lambda F: sp.diags(1 / np.sqrt(np.asarray(F.multiply(F).sum(1)).ravel() + 1e-9)) @ F
    return (n(FM) @ n(FH).T).toarray().astype(np.float32)


def ollama(prompt, n=8):
    body = json.dumps({"model": LLM, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": n}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["response"]


def describe(o, c, parents):
    ps = [o.label[t] for h, r, t in parents.get(c, [])][:4]
    syn = o.synonyms[c][:3]
    return o.label[c] + (f" (synonyms: {', '.join(syn)})" if syn else "") + (f" [parents: {', '.join(ps)}]" if ps else "")


def main():
    M, H = oaei.load("mouse"), oaei.load("human")
    ref = oaei.reference()
    mi = {c: i for i, c in enumerate(M.classes)}
    hi = {c: i for i, c in enumerate(H.classes)}
    lab_h: dict[str, list[str]] = {}
    for c, l in H.label.items():
        lab_h.setdefault(l, []).append(c)
    anchors = [(c, lab_h[l][0]) for c, l in M.label.items() if l in lab_h and len(lab_h[l]) == 1]
    R = set(ref)
    res = {"mouse_classes": len(M.classes), "human_classes": len(H.classes), "reference": len(ref),
           "auto_anchors(exact label)": {"n": len(anchors), "precision": sum(p in R for p in anchors) / len(anchors),
                                         "recall": sum(p in R for p in anchors) / len(R)}}
    nontriv = [p for p in ref if M.label[p[0]] != H.label[p[1]]]
    anchored_h = {b for _, b in anchors}
    cand_cols = np.array([hi[c] for c in H.classes if c not in anchored_h])  # unmatched human classes
    col_pos = {c: j for j, c in enumerate(cand_cols)}
    eval_pairs = [(a, b) for a, b in nontriv if hi[b] in col_pos]
    rows = np.array([mi[a] for a, _ in eval_pairs])
    gold = np.array([col_pos[hi[b]] for _, b in eval_pairs])
    res["nontrivial_eval_pairs"] = len(eval_pairs)
    res["candidates_per_query"] = len(cand_cols)
    log(res)
    E = lambda S: rank_metrics(S[rows][:, cand_cols], rows, cand_cols, gold)
    out, sims = {}, {}

    # --- structure only ---
    Mt = [(f"M:{h}", f"M:{r}", f"M:{t}") for h, r, t in M.edges]
    Ht = [(f"H:{h}", f"H:{r}", f"H:{t}") for h, r, t in H.edges]
    tfM, EM = train_struct(Mt, 0)
    tfH, EH = train_struct(Ht, 1)
    XM = EM[[tfM.entity_to_id[f"M:{c}"] if f"M:{c}" in tfM.entity_to_id else 0 for c in M.classes]]
    XH = EH[[tfH.entity_to_id[f"H:{c}"] if f"H:{c}" in tfH.entity_to_id else 0 for c in H.classes]]
    sims["structure: independent, no anchors"] = cos(XM, XH)
    A_m = np.array([mi[a] for a, _ in anchors]); A_h = np.array([hi[b] for _, b in anchors])
    muM, muH = XM[A_m].mean(0), XH[A_h].mean(0)
    u, _, vt = np.linalg.svd((XH[A_h] - muH).T @ (XM[A_m] - muM))
    sims["structure: independent + Procrustes(auto anchors)"] = cos(XM - muM, (XH - muH) @ (u @ vt))
    amap = {b: a for a, b in anchors}
    Jt = Mt + [(f"M:{amap[h]}" if h in amap else f"H:{h}", f"H:{r}", f"M:{amap[t]}" if t in amap else f"H:{t}") for h, r, t in H.edges]
    tfJ, EJ = train_struct(Jt, 0)
    get = lambda pre, c: EJ[tfJ.entity_to_id.get(f"{pre}:{c}", tfJ.entity_to_id.get(f"M:{amap.get(c, '')}", 0))]
    JM = np.stack([get("M", c) for c in M.classes]); JH = np.stack([get("H", c) for c in H.classes])
    sims["structure: joint RotatE (auto anchors)"] = cos(JM, JH)
    sims["structure: anchor-neighbourhood (no embedding)"] = anchor_neigh(M, H, anchors)
    log("structure done")

    # --- text ---
    sims["lexical: char 3-4gram TF-IDF (+syn)"] = lexical_sim(M, H)
    for model in ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5"):
        sims[f"text: {model.split('/')[-1]} (label only)"] = text_sim(model, M, H, with_syn=False)
        sims[f"text: {model.split('/')[-1]} (+synonyms)"] = text_sim(model, M, H, with_syn=True)
        log("text", model)
    best_text = "text: bge-large-en-v1.5 (+synonyms)"
    sims["hybrid: bge-large + joint RotatE"] = zs(sims[best_text]) + 0.5 * zs(sims["structure: joint RotatE (auto anchors)"])
    sims["hybrid: bge-large + anchor-neighbourhood"] = zs(sims[best_text]) + 0.5 * zs(sims["structure: anchor-neighbourhood (no embedding)"])
    sims["hybrid: bge-large + joint + anchor-neigh + lexical"] = (
        zs(sims[best_text]) + 0.5 * zs(sims["structure: joint RotatE (auto anchors)"])
        + 0.5 * zs(sims["structure: anchor-neighbourhood (no embedding)"]) + 0.5 * zs(sims["lexical: char 3-4gram TF-IDF (+syn)"]))
    for k, S in sims.items():
        out[k] = E(S)
        log(f"{k:58s} H1={out[k]['Hits@1']:.3f} H10={out[k]['Hits@10']:.3f} MRR={out[k]['MRR']:.3f}")
    res["nontrivial_ranking"] = out
    (RESULTS / "exp5_oaei_anatomy.json").write_text(json.dumps(res, indent=2))

    # --- LLM re-rank of the best hybrid's top-10 on the non-trivial pairs ---
    best = "hybrid: bge-large + joint + anchor-neigh + lexical"
    parM, parH = {}, {}
    for h, r, t in M.edges:
        parM.setdefault(h, []).append((h, r, t))
    for h, r, t in H.edges:
        parH.setdefault(h, []).append((h, r, t))
    S = sims[best][rows][:, cand_cols]
    hits_base = hits_llm = in_top = 0
    ex = []
    for i, (a, b) in enumerate(eval_pairs):
        top = list(np.argsort(-S[i])[:10])
        hits_base += top[0] == gold[i]
        in_top += gold[i] in top
        opts = "\n".join(f"{j+1}. {describe(H, H.classes[cand_cols[c]], parH)}" for j, c in enumerate(top))
        p = (f"Mouse anatomy term: {describe(M, a, parM)}\n\nWhich human anatomy term below denotes the SAME anatomical "
             f"entity? If none does, answer 0.\n{opts}\n\nAnswer with the number only.")
        try:
            ans = ollama(p)
            k = int("".join(ch for ch in ans if ch.isdigit())[:2] or 0) - 1
        except Exception:  # noqa: BLE001
            k = -1
        ok = 0 <= k < len(top) and top[k] == gold[i]
        hits_llm += ok
        if len(ex) < 25:
            ex.append({"mouse": M.label[a], "human_gold": H.label[b],
                       "hybrid_top1": H.label[H.classes[cand_cols[top[0]]]],
                       "llm": H.label[H.classes[cand_cols[top[k]]]] if 0 <= k < len(top) else "none", "llm_ok": bool(ok)})
    res["llm_rerank_nontrivial"] = {"model": LLM, "n": len(eval_pairs), "gold_in_top10": in_top / len(eval_pairs),
                                    "hybrid_Hits@1": hits_base / len(eval_pairs), "llm_Hits@1": hits_llm / len(eval_pairs),
                                    "examples": ex}
    log("llm rerank", {k: v for k, v in res["llm_rerank_nontrivial"].items() if k != "examples"})
    (RESULTS / "exp5_oaei_anatomy.json").write_text(json.dumps(res, indent=2))

    # --- full alignment, P/R/F1 vs the OAEI reference ---
    def prf(pairs):
        P = set(pairs)
        tp = len(P & R)
        p, r = tp / max(len(P), 1), tp / len(R)
        return {"pairs": len(P), "precision": p, "recall": r, "F1": 2 * p * r / max(p + r, 1e-9)}

    align = {"exact labels only": prf(anchors)}
    unm = [c for c in M.classes if c not in {a for a, _ in anchors}]
    um_rows = np.array([mi[c] for c in unm])
    for name in (best_text, best):
        Sx = sims[name][um_rows][:, cand_cols]
        col_best = Sx.argmax(0)
        for tau in (0.0, 1.0, 2.0, 3.0, 4.0) if "hybrid" in name else (0.80, 0.85, 0.90, 0.95):
            extra = []
            for i, c in enumerate(unm):
                j = int(Sx[i].argmax())
                if col_best[j] == i and Sx[i, j] >= tau:  # mutual best + threshold
                    extra.append((c, H.classes[cand_cols[j]]))
            align[f"exact + {name} mutual-best >= {tau}"] = prf(anchors + extra)
    # LLM verification of mutual-best candidates from the best hybrid
    Sx = sims[best][um_rows][:, cand_cols]
    col_best = Sx.argmax(0)
    cands = [(c, H.classes[cand_cols[int(Sx[i].argmax())]]) for i, c in enumerate(unm) if col_best[int(Sx[i].argmax())] == i]
    kept = []
    for a, b in cands:
        p = (f"Mouse anatomy term: {describe(M, a, parM)}\nHuman anatomy term: {describe(H, b, parH)}\n\n"
             "Do these denote the same anatomical entity (an equivalence mapping between the two ontologies)? Answer yes or no.")
        try:
            if ollama(p, n=3).strip().lower().startswith("y"):
                kept.append((a, b))
        except Exception:  # noqa: BLE001
            pass
    align["exact + hybrid mutual-best + LLM yes/no verification"] = prf(anchors + kept)
    align["_llm_verified_candidates"] = {"candidates": len(cands), "kept": len(kept)}
    res["alignment_PRF"] = align
    for k, v in align.items():
        log(k, v)
    (RESULTS / "exp5_oaei_anatomy.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
