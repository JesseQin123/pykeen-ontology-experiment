"""Experiment 17 - LLMs on top of structure.

A. WN18RR, full test set (5,848 queries): an LLM re-ranks the top-20 of RRF(AnyBURL, RotatE, TransE).
   Filtered ranks: the LLM's pick moves to rank 1, the rest keep the RRF order; gold outside the top-20 keeps its rank.
B. GO suggestion pipeline: for 1,000 random terms, an ensemble proposes 5 new parents each (never already known or
   implied); three LLM judges vote; we measure how many accepted suggestions are true in the 2026 release
   (the parent became an ancestor), against the un-filtered top-1.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from kgv import anyburl, data, go, kge, remote  # noqa: E402

OUT = ROOT / "results" / "exp17_llm_on_structure.json"
MODELS = ("gpt-6.1-sol", "gemini-4-argon-high", "claude-opus-5-5")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(res):
    OUT.write_text(json.dumps(res, indent=2))


def first_int(s):
    m = re.search(r"\d+", s or "")
    return int(m.group(0)) if m else -1


def is_yes(s):
    return re.sub(r"^[\W_]+", "", s or "").lower().startswith("y")


def part_a(res, k=20):
    from exp13_ensembles import anyburl_scores, rrf
    from exp16_llm_blind_rating import PHRASE
    from kgv import wordnet
    kg = data.load("wn18rr")
    wordnet.resolve(kg)
    R = kg.num_relations
    A = anyburl_scores(kg, anyburl.parse(ROOT / "runs/anyburl_wn18rr/preds-100"))
    rot = kge.score_fn(torch.load(ROOT / "runs/wn18rr_RotatE_s0.pt", weights_only=False, map_location="mps"), R)
    tra = kge.score_fn(torch.load(ROOT / "runs/wn18rr_TransE_s0.pt", weights_only=False, map_location="mps"), R)
    f = rrf(A, rot, tra)
    q = data.as_queries(kg.test, R)
    known = kg.known_tails()
    desc = lambda e: (lambda s: f"{s.lemmas()[0].name().replace('_', ' ')} ({s.definition()})")(wordnet.synset(kg.entity_labels[e]))
    base_rank, tops, gold_pos, prompts = [], [], [], []
    for s in range(0, len(q), 256):
        b = q[s : s + 256]
        sc = f(b[:, 0], b[:, 1])
        for j, (h, d, t) in enumerate(b):
            h, d, t = int(h), int(d), int(t)
            row = sc[j].copy()
            row[[x for x in known[(h, d)] if x != t]] = -np.inf
            true = row[t]
            base_rank.append((row > true).sum() + 1 + ((row == true).sum() - 1) / 2)
            top = [int(c) for c in np.argsort(-row)[:k]]
            tops.append(top)
            gold_pos.append(top.index(t) + 1 if t in top else 0)
            rel = kg.relation_labels[d % R]
            if d < R:
                claim = PHRASE[rel].format(h=desc(h), t="X")
            else:
                claim = PHRASE[rel].format(h="X", t=desc(h))
            opts = "\n".join(f"{i+1}. {desc(c)}" for i, c in enumerate(top))
            prompts.append(f"WordNet lexical semantics. Which candidate X makes this claim true?\nClaim: {claim}\n\n"
                           f"Candidates:\n{opts}\n\nAnswer with the number only.")
    base_rank = np.asarray(base_rank, float)
    gold_pos = np.asarray(gold_pos)
    summ = lambda r: {"MRR": float(np.mean(1 / r)), "Hits@1": float(np.mean(r <= 1)), "Hits@10": float(np.mean(r <= 10)), "n": len(r)}
    res.setdefault("A_wn18rr_rerank", {})["RRF(AnyBURL,RotatE,TransE) alone"] = summ(base_rank)
    res["A_wn18rr_rerank"]["_note"] = "gemini-4-argon-high skipped: relay returned HTTP 429 (quota exhausted)"
    res["A_wn18rr_rerank"]["gold_in_top20"] = float(np.mean(gold_pos > 0))
    save(res)
    for m in MODELS:
        if f"+ {m} re-rank" in res["A_wn18rr_rerank"]:
            continue  # already scored in an earlier run
        picks = [first_int(o) for o in remote.chat_many(m, prompts, max_tokens=8, workers=24)]
        new = base_rank.copy()
        for i, p in enumerate(picks):
            g = gold_pos[i]
            if g == 0:
                continue
            if p == g:
                new[i] = 1
            elif 1 <= p <= k:
                new[i] = g + 1 if p > g else g  # the pick jumps ahead of everything; gold shifts down by one if it was above it
        res["A_wn18rr_rerank"][f"+ {m} re-rank"] = summ(new)
        log("A", m, res["A_wn18rr_rerank"][f"+ {m} re-rank"]); save(res)


def part_b(res, n_terms=1000, per_term=5, seed=1):
    from exp4_go_temporal import TextScorer
    from exp13_ensembles import rrf
    kg, _, _, _, terms = go.temporal("2020-01-01", "2026-08-05")
    _, e26 = go.parse("2026-08-05")
    anc26 = go.closure_is_a(e26)
    lab = kg.entity_labels
    eid = {e: i for i, e in enumerate(lab)}
    old_anc = go.closure_is_a([(lab[h], kg.relation_labels[r], lab[t]) for h, r, t in kg.train])
    tag = "2020-01-01_2026-08-05"
    rot = kge.score_fn(torch.load(ROOT / f"runs/go_{tag}_RotatE_s0.pt", weights_only=False, map_location="mps"), kg.num_relations)
    tra = kge.score_fn(torch.load(ROOT / f"runs/go_{tag}_TransE_s0.pt", weights_only=False, map_location="mps"), kg.num_relations)
    txt = TextScorer(kg, terms, model="BAAI/bge-large-en-v1.5").score
    ens = rrf(rot, tra, txt)
    isa = kg.relation_labels.index("is_a")
    known = kg.known_tails()
    rng = np.random.default_rng(seed)
    heads = rng.choice(np.unique(kg.train[kg.train[:, 1] == isa][:, 0]), n_terms, replace=False)
    sugg = []
    for s in range(0, len(heads), 128):
        b = heads[s : s + 128]
        sc = ens(b, np.full(len(b), isa))
        for j, h in enumerate(b):
            excl = set(known.get((int(h), isa), ())) | {eid[a] for a in old_anc.get(lab[h], ())} | {int(h)}
            row = sc[j].copy(); row[list(excl)] = -np.inf
            for rank, c in enumerate(np.argsort(-row)[:per_term]):
                sugg.append({"child": lab[h], "parent": lab[int(c)], "rank": rank + 1,
                             "true_2026": lab[int(c)] in anc26.get(lab[h], set())})
    random.Random(seed).shuffle(sugg)
    prompts = []
    for it in sugg:
        c, p = terms[it["child"]], terms[it["parent"]]
        prompts.append(f"Gene Ontology question.\nTerm A: \"{c.name}\" ({c.namespace}). Definition: {c.definition}\n"
                       f"Term B: \"{p.name}\" ({p.namespace}). Definition: {p.definition}\n\n"
                       "Is A a subtype of B (A is_a B), i.e. is every instance of A necessarily an instance of B? Answer yes or no.")
    votes = {}
    for m in MODELS:
        votes[m] = [is_yes(o) for o in remote.chat_many(m, prompts, max_tokens=8, workers=24)]
        log("B judged", m, np.mean(votes[m]))
    maj = [sum(votes[m][i] for m in MODELS) >= 2 for i in range(len(sugg))]
    truth = np.array([it["true_2026"] for it in sugg])
    rank = np.array([it["rank"] for it in sugg])
    maj = np.array(maj)
    out = {"suggestions": len(sugg), "terms": n_terms,
           "true_in_2026_all_top5": float(truth.mean()), "true_in_2026_top1_only": float(truth[rank == 1].mean()),
           "LLM_majority_accept_rate": float(maj.mean()),
           "true_in_2026_among_accepted": float(truth[maj].mean()) if maj.any() else None,
           "true_in_2026_among_rejected": float(truth[~maj].mean()) if (~maj).any() else None,
           "recall_of_true_by_LLM_filter": float(maj[truth].mean()) if truth.any() else None,
           "terms_with_any_accepted": int(len({it["child"] for it, a in zip(sugg, maj) if a}))}
    for m in MODELS:
        v = np.array(votes[m])
        out[f"accept_rate_{m}"] = float(v.mean())
        out[f"true_among_accepted_{m}"] = float(truth[v].mean()) if v.any() else None
    res["B_go_suggestion_pipeline"] = out
    res["B_examples_accepted_not_in_2026"] = [{"child": terms[it["child"]].name, "parent": terms[it["parent"]].name}
                                              for it, a in zip(sugg, maj) if a and not it["true_2026"]][:40]
    log("B", out); save(res)


def main():
    global MODELS
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    args = sys.argv[1:]
    if "--models" in args:  # e.g. --models claude-opus-5-5 (Gemini's relay quota ran out: HTTP 429)
        i = args.index("--models"); MODELS = tuple(args[i + 1].split(",")); args = args[:i] + args[i + 2:]
    parts = args or ["A", "B"]
    if "B" in parts:
        part_b(res)
    if "A" in parts:
        part_a(res)
    log("done")


if __name__ == "__main__":
    main()
