"""Experiment 3 - Do graph embeddings "naturally" align two ontologies that use different terms?

Controlled setup (ground truth known):
  * Start from one real graph. Build two views A and B, each keeping an independent random 70% of the triples,
    so they overlap but are not identical (like two teams modelling the same domain).
  * Rename every entity and relation in B (random permutation; B never shares a single identifier with A).
  * Gold alignment = entities present in both views. A fraction s of gold pairs is given as "seed" anchors
    (e.g. a few hand-mapped concepts); the rest is the test set.

Methods
  independent+cosine   train A and B separately, match by cosine. Tests the "naturally aligns" hypothesis.
  independent+Procrustes  same embeddings, rotate B onto A using the seed anchors (MUSE/Procrustes style).
  joint (shared anchors)  train one model on A u B where seed anchors share one embedding (MTransE-style).
  anchor-neighbourhood  NO embeddings: match by which seed anchors sit in each entity's 1-2 hop neighbourhood.
  relation alignment    from the joint model: does A's renamed relation land next to B's renamed twin?

Exp 3b (WordNet only): B uses a *different synonym* as each concept's label (car vs automobile), so we can also
compare label-only matching (char n-grams, sentence embeddings) and structure+text hybrids.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from pykeen.triples import TriplesFactory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import data, kge  # noqa: E402

RESULTS = ROOT / "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_views(kg, keep=0.7, seed=0):
    rng = np.random.default_rng(seed)
    T = kg.train
    A = T[rng.random(len(T)) < keep]
    B = T[rng.random(len(T)) < keep]
    gold = np.intersect1d(np.unique(A[:, [0, 2]]), np.unique(B[:, [0, 2]]))
    perm_e = rng.permutation(kg.num_entities)
    perm_r = rng.permutation(kg.num_relations)
    return A, B, gold, perm_e, perm_r


def labeled(A, B, perm_e, perm_r, anchors=frozenset(), which="AB"):
    """String-labelled triples. B entities are renamed via perm_e; B anchors are collapsed onto their A name."""
    rows = []
    if "A" in which:
        rows += [(f"A:{h}", f"A:r{r}", f"A:{t}") for h, r, t in A]
    if "B" in which:
        nb = lambda e: f"A:{e}" if e in anchors else f"B:{perm_e[e]}"
        rows += [(nb(h), f"B:r{perm_r[r]}", nb(t)) for h, r, t in B]
    return np.array(rows, dtype=str)


def fit(triples, seed, epochs, device):
    tf = TriplesFactory.from_labeled_triples(triples)
    m, info = kge.train(None, "RotatE", tf=tf, seed=seed, device=device, epochs=epochs, lr=1e-2)
    return tf, kge.entity_embeddings(m), kge.relation_embeddings(m), info


def rows_of(tf, labels):
    return np.array([tf.entity_to_id[x] for x in labels])


def align_scores(XA, XB, chunk=2048):
    """Rank of the true counterpart (row i of XB) for each row i of XA under cosine similarity."""
    a = torch.nn.functional.normalize(torch.as_tensor(XA), dim=1)
    b = torch.nn.functional.normalize(torch.as_tensor(XB), dim=1)
    ranks = []
    for s in range(0, len(a), chunk):
        sim = a[s : s + chunk] @ b.T
        true = sim[torch.arange(len(sim)), torch.arange(s, s + len(sim))]
        ranks.append((sim > true[:, None]).sum(1) + 1)
    return torch.cat(ranks).numpy().astype(float)


def rank_from_sim(sim):
    true = np.diag(sim)
    return (sim > true[:, None]).sum(1) + 1 + ((sim == true[:, None]).sum(1) - 1) / 2


def metrics(r):
    return {"Hits@1": float(np.mean(r <= 1)), "Hits@10": float(np.mean(r <= 10)), "MRR": float(np.mean(1 / r)),
            "n": int(len(r))}


def procrustes(XA_seed, XB_seed):
    """Orthogonal W minimising ||XB W - XA|| after centring (embeddings are only defined up to such maps)."""
    u, _, vt = np.linalg.svd(XB_seed.T @ XA_seed)
    return u @ vt


def anchor_neighbourhood(n, A, B, perm_e, anchors, test):
    """Pure graph-search baseline: cosine between bags of seed anchors within 2 hops (1-hop weighted x2)."""
    anchors = np.array(sorted(anchors))
    def feats(T, ids_of):
        nn = max(T[:, [0, 2]].max(), n) + 1
        adj = sp.csr_matrix((np.ones(len(T)), (T[:, 0], T[:, 2])), shape=(nn, nn))
        adj = ((adj + adj.T) > 0).astype(np.float32)
        two = ((adj @ adj) > 0).astype(np.float32)
        M = (2 * adj + two).tocsr()
        return M[ids_of(test)][:, ids_of(anchors)]
    FA = feats(A, lambda e: e)
    Bm = np.stack([perm_e[B[:, 0]], B[:, 1], perm_e[B[:, 2]]], 1)
    FB = feats(Bm, lambda e: perm_e[e])
    norm = lambda F: sp.diags(1 / np.sqrt(np.asarray(F.multiply(F).sum(1)).ravel() + 1e-9)) @ F
    sim = (norm(FA) @ norm(FB).T).toarray()
    return rank_from_sim(sim), sim


def text_labels(kg, entities):
    """A uses a concept's first synonym, B a different synonym (only concepts with >= 2 distinct lemmas)."""
    from kgv import wordnet
    wordnet.resolve(kg)
    keep, la, lb = [], [], []
    for e in entities:
        s = wordnet.synset(kg.entity_labels[e])
        names = list(dict.fromkeys(l.name().replace("_", " ").lower() for l in s.lemmas()))
        if len(names) >= 2:
            keep.append(e); la.append(names[0]); lb.append(names[1])
    return np.array(keep), la, lb


def zs(sim):
    return (sim - sim.mean(1, keepdims=True)) / (sim.std(1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wn18rr")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--fractions", default="0.01,0.05,0.1,0.2,0.3")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    fracs = [float(x) for x in args.fractions.split(",")]

    kg = data.load(args.dataset)
    A, B, gold, perm_e, perm_r = make_views(kg)
    res = {"dataset": args.dataset, "views": {"A_triples": len(A), "B_triples": len(B), "gold_pairs": len(gold)}, "results": {}}
    log(res["views"])
    rng = np.random.default_rng(1)
    order = rng.permutation(gold)
    lab_a = lambda es: [f"A:{e}" for e in es]
    lab_b = lambda es: [f"B:{perm_e[e]}" for e in es]

    # independent spaces
    tfA, EA, _, _ = fit(labeled(A, B, perm_e, perm_r, which="A"), 0, args.epochs, args.device)
    tfB, EB, _, _ = fit(labeled(A, B, perm_e, perm_r, which="B"), 1, args.epochs, args.device)
    log("independent models trained")

    text = {} if args.dataset == "wn18rr" else None
    for frac in [0.0] + fracs:
        n_seed = int(round(frac * len(gold)))
        seeds, test = order[:n_seed], order[n_seed:]
        out = {"n_seed": n_seed}
        XA, XB = EA[rows_of(tfA, lab_a(test))], EB[rows_of(tfB, lab_b(test))]
        if frac == 0.0:
            out["independent+cosine"] = metrics(align_scores(XA, XB))
            out["random_expectation_Hits@1"] = 1 / len(test)
        else:
            SA, SB = EA[rows_of(tfA, lab_a(seeds))], EB[rows_of(tfB, lab_b(seeds))]
            muA, muB = SA.mean(0), SB.mean(0)
            W = procrustes(SA - muA, SB - muB)
            out["independent+Procrustes"] = metrics(align_scores(XA - muA, (XB - muB) @ W))
            out["seed_fit_Hits@1"] = metrics(align_scores(SA - muA, (SB - muB) @ W))["Hits@1"]

            r_nb, sim_nb = anchor_neighbourhood(kg.num_entities, A, B, perm_e, set(seeds.tolist()), test)
            out["anchor-neighbourhood (no embedding)"] = metrics(r_nb)

            tfJ, EJ, RJ, info = fit(labeled(A, B, perm_e, perm_r, anchors=set(seeds.tolist())), 0, args.epochs,
                                    args.device)
            JA, JB = EJ[rows_of(tfJ, lab_a(test))], EJ[rows_of(tfJ, lab_b(test))]
            r_joint = align_scores(JA, JB)
            out["joint (shared anchors)"] = metrics(r_joint)
            # relation alignment: RotatE relations are phases; compare on the unit circle
            ra = np.array([tfJ.relation_to_id[f"A:r{r}"] for r in range(kg.num_relations)])
            rb = np.array([tfJ.relation_to_id[f"B:r{perm_r[r]}"] for r in range(kg.num_relations)])
            out["relation_alignment_Hits@1(joint)"] = metrics(align_scores(RJ[ra], RJ[rb]))["Hits@1"]
            out["relations_total"] = kg.num_relations

            if text is not None and frac in (0.05, 0.3):
                keep, la, lb = text_labels(kg, test)
                pos = {int(e): i for i, e in enumerate(test)}
                idx = np.array([pos[int(e)] for e in keep])
                aj = torch.nn.functional.normalize(torch.as_tensor(JA[idx]), dim=1)
                bj = torch.nn.functional.normalize(torch.as_tensor(JB[idx]), dim=1)
                s_struct = (aj @ bj.T).numpy()
                from sklearn.feature_extraction.text import TfidfVectorizer
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3)).fit(la + lb)
                s_lex = (vec.transform(la) @ vec.transform(lb).T).toarray()
                from sentence_transformers import SentenceTransformer
                st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
                ea = st.encode(la, normalize_embeddings=True, batch_size=256)
                eb = st.encode(lb, normalize_embeddings=True, batch_size=256)
                s_sem = ea @ eb.T
                text[f"seed={frac}"] = {
                    "n_concepts": int(len(keep)),
                    "structure (joint RotatE)": metrics(rank_from_sim(s_struct)),
                    "lexical (char 3-gram)": metrics(rank_from_sim(s_lex)),
                    "text embedding (MiniLM)": metrics(rank_from_sim(s_sem)),
                    "hybrid structure+MiniLM": metrics(rank_from_sim(zs(s_struct) + zs(s_sem))),
                    "examples_label_pairs": list(zip(la[:15], lb[:15])),
                }
                log("text", frac, {k: v for k, v in text[f"seed={frac}"].items() if k != "examples_label_pairs"})
        res["results"][f"seed={frac}"] = out
        log(frac, {k: v for k, v in out.items()})
        (RESULTS / f"exp3_alignment_{args.dataset}.json").write_text(json.dumps({**res, "text": text}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
