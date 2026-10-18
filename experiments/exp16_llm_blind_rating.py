"""Experiment 16 - Blind rating of suggested links by three frontier LLM judges (a scalable stand-in for expert review).

The open-world problem: a suggestion absent from the graph may still be true. Expert review is the gold standard;
here three independent LLM judges (Claude Opus 5.5, GPT-6.1, Gemini 4 Argon) rate shuffled suggestions without knowing their source.
Calibration controls are mixed in:
  * positive controls: edges known to be true (held-out WN18RR test edges; is_a links GO curators added by 2026)
  * negative controls: random candidates
For GO we also have an objective check: is the suggested parent an ancestor of the term in the 2026 release?
Reports per source: each judge's yes-rate, majority and unanimous yes-rates, pairwise agreement (Cohen's kappa),
and objective truth where known.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from kgv import data, go, kge, remote  # noqa: E402

OUT = ROOT / "results" / "exp16_llm_blind_rating.json"
JUDGES = ("claude-opus-5-5", "gpt-6.1-sol", "gemini-4-argon-high")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def is_yes(s):
    import re
    return re.sub(r"^[\W_]+", "", s or "").lower().startswith("y")


def judge(items, prompt_fn):
    """items: list of dicts with a 'key'; returns {judge: {key: bool}} (deduplicated prompts)."""
    uniq = {}
    for it in items:
        uniq.setdefault(it["key"], it)
    keys = list(uniq)
    prompts = [prompt_fn(uniq[k]) for k in keys]
    out = {}
    for j in JUDGES:
        ans = remote.chat_many(j, prompts, max_tokens=8, workers=16)
        out[j] = {k: is_yes(a) for k, a in zip(keys, ans)}
        log("judged", j, len(keys), "yes-rate", np.mean(list(out[j].values())))
    return out


def summarize(items, verdicts, extra_fields=()):
    res = {}
    sources = sorted({it["source"] for it in items})
    for s in sources:
        its = [it for it in items if it["source"] == s]
        row = {"n": len(its)}
        for j in JUDGES:
            row[f"yes_{j}"] = float(np.mean([verdicts[j][it["key"]] for it in its]))
        row["yes_all"] = float(np.mean([all(verdicts[j][it["key"]] for j in JUDGES) for it in its]))
        row["yes_majority"] = float(np.mean([sum(verdicts[j][it["key"]] for j in JUDGES) * 2 > len(JUDGES) for it in its]))
        for f in extra_fields:
            vals = [it[f] for it in its if it.get(f) is not None]
            if vals:
                row[f] = float(np.mean(vals))
        res[s] = row
    allk = sorted({it["key"] for it in items})
    from itertools import combinations
    res["_agreement_kappa"] = {f"{a} vs {b}": kappa([verdicts[a][k] for k in allk], [verdicts[b][k] for k in allk])
                               for a, b in combinations(JUDGES, 2)}
    return res


# ------------------------------------------------------------------ GO
def go_part(n_terms=1000, seed=0):
    from exp4_go_temporal import TextScorer, zfuse
    kg, _, kinds, _, terms = go.temporal("2020-01-01", "2026-08-05")
    _, e26 = go.parse("2026-08-05")
    anc26 = go.closure_is_a(e26)
    lab = kg.entity_labels
    old_anc = go.closure_is_a([(lab[h], kg.relation_labels[r], lab[t]) for h, r, t in kg.train])
    eid = {e: i for i, e in enumerate(lab)}
    tag = "2020-01-01_2026-08-05"
    rot = kge.score_fn(torch.load(ROOT / f"runs/go_{tag}_RotatE_s0.pt", weights_only=False, map_location="mps"), kg.num_relations)
    tra = kge.score_fn(torch.load(ROOT / f"runs/go_{tag}_TransE_s0.pt", weights_only=False, map_location="mps"), kg.num_relations)
    txt = TextScorer(kg, terms, model="BAAI/bge-large-en-v1.5").score
    srcs = {"Hybrid(RotatE+Text)": zfuse(rot, txt), "RotatE": rot, "TransE": tra, "Text(bge-large)": txt}
    isa = kg.relation_labels.index("is_a")
    known = kg.known_tails()
    rng = np.random.default_rng(seed)
    has_parent = np.unique(kg.train[kg.train[:, 1] == isa][:, 0])
    heads = rng.choice(has_parent, n_terms, replace=False)
    items = []

    def add(h, c, source):
        child, parent = lab[h], lab[c]
        items.append({"key": f"{child}|{parent}", "source": source, "child": child, "parent": parent,
                      "true_in_2026_closure": parent in anc26.get(child, set())})

    for name, fn in srcs.items():
        for s in range(0, len(heads), 128):
            b = heads[s : s + 128]
            sc = np.asarray(fn(b, np.full(len(b), isa)), dtype=np.float64)
            for j, h in enumerate(b):
                excl = set(known.get((int(h), isa), ())) | {eid[a] for a in old_anc.get(lab[h], ())} | {int(h)}
                sc[j, list(excl)] = -np.inf
                add(int(h), int(np.argmax(sc[j])), name)
    for h in heads:  # negative control
        excl = {eid[a] for a in old_anc.get(lab[h], ())} | {int(h)}
        c = int(rng.integers(kg.num_entities))
        while c in excl:
            c = int(rng.integers(kg.num_entities))
        add(int(h), c, "control: random candidate")
    novel_isa = [i for i in range(len(kg.test)) if kinds[i] == "novel" and kg.test[i, 1] == isa]
    for i in rng.permutation(novel_isa)[:n_terms]:  # positive control: curator additions 2020->2026
        h, _, t = kg.test[i]
        add(int(h), int(t), "control: curator-added 2020-2026 (true)")
    for i in rng.permutation(np.where(kg.train[:, 1] == isa)[0])[:300]:  # positive control: existing edges
        h, _, t = kg.train[i]
        add(int(h), int(t), "control: existing 2020 edge (true)")
    random.Random(seed).shuffle(items)

    def prompt(it):
        c, p = terms[it["child"]], terms[it["parent"]]
        return (f"Gene Ontology question.\nTerm A: \"{c.name}\" ({c.namespace}). Definition: {c.definition}\n"
                f"Term B: \"{p.name}\" ({p.namespace}). Definition: {p.definition}\n\n"
                "Is A a subtype of B (A is_a B), i.e. is every instance of A necessarily an instance of B? Answer yes or no.")
    verdicts = judge(items, prompt)
    res = summarize(items, verdicts, extra_fields=("true_in_2026_closure",))
    ex = [{"source": it["source"], "child": terms[it["child"]].name, "parent": terms[it["parent"]].name,
           **{j: verdicts[j][it["key"]] for j in JUDGES}, "true_2026": it["true_in_2026_closure"]}
          for it in items if not it["source"].startswith("control")][:60]
    return {"summary": res, "examples": ex}


# ------------------------------------------------------------------ WN18RR
PHRASE = {
    "_hypernym": "\"{h}\" is a kind of \"{t}\"",
    "_instance_hypernym": "\"{h}\" is an instance of \"{t}\"",
    "_has_part": "\"{h}\" has \"{t}\" as a part",
    "_member_meronym": "\"{h}\" has \"{t}\" as a member",
    "_derivationally_related_form": "\"{h}\" and \"{t}\" are derivationally related word forms",
    "_also_see": "\"{h}\" is a closely related sense (see also) of \"{t}\"",
    "_verb_group": "\"{h}\" and \"{t}\" are closely related senses of the same verb",
    "_similar_to": "\"{h}\" is similar in meaning to \"{t}\"",
    "_synset_domain_topic_of": "\"{h}\" belongs to the topic domain \"{t}\"",
    "_member_of_domain_region": "\"{t}\" is a term used in, or specific to, the region \"{h}\"",
    "_member_of_domain_usage": "\"{t}\" has the usage label \"{h}\"",
}


def wn_part(n=1500, seed=0):
    from kgv import anyburl, wordnet
    kg = data.load("wn18rr")
    ver = wordnet.Verifier(kg)
    R = kg.num_relations
    rot = kge.score_fn(torch.load(ROOT / "runs/wn18rr_RotatE_s0.pt", weights_only=False, map_location="mps"), R)
    preds = anyburl.parse(ROOT / "runs/anyburl_wn18rr/preds-100")
    ab = {}
    for (h, r, t), (hc, tc) in preds.items():
        ab.setdefault((h, r), dict(tc)); ab.setdefault((t, r + R), dict(hc))
    train_known = {}
    for h, r, t in kg.train:
        train_known.setdefault((int(h), int(r)), set()).add(int(t))
        train_known.setdefault((int(t), int(r) + R), set()).add(int(h))
    held = kg.known_tails()
    q = data.as_queries(kg.test, R)
    rng = np.random.default_rng(seed)
    uq = np.unique(q[:, :2], axis=0)
    uq = uq[rng.permutation(len(uq))[:n]]
    items = []

    def add(h, d, c, source):
        trip = (h, d, c) if d < R else (c, d - R, h)
        v = "held-out" if c in held.get((h, d), ()) else ver.verdict(*trip)
        items.append({"key": f"{trip[0]}|{trip[1]}|{trip[2]}", "source": source, "trip": trip, "verifier": v,
                      "true_by_wordnet": v in ("held-out", "direct", "entailed")})

    sc = rot(uq[:, 0], uq[:, 1]).astype(np.float64)
    for j, (h, d) in enumerate(uq):
        h, d = int(h), int(d)
        sc[j, list(train_known.get((h, d), ()))] = -np.inf
        sc[j, h] = -np.inf
        add(h, d, int(np.argmax(sc[j])), "RotatE top-1")
        cands = [(e, c) for e, c in sorted(ab.get((h, d), {}).items(), key=lambda x: -x[1]) if e not in train_known.get((h, d), ()) and e != h]
        if cands:
            add(h, d, cands[0][0], "AnyBURL top-1")
        add(h, d, int(rng.integers(kg.num_entities)), "control: random candidate")
    for h, r, t in kg.test[rng.permutation(len(kg.test))[:600]]:
        add(int(h), int(r), int(t), "control: held-out test edge (true)")
    random.Random(seed).shuffle(items)

    def prompt(it):
        h, r, t = it["trip"]
        sh, st = wordnet.synset(kg.entity_labels[h]), wordnet.synset(kg.entity_labels[t])
        nh = f"{sh.lemmas()[0].name().replace('_', ' ')} ({sh.definition()})"
        nt = f"{st.lemmas()[0].name().replace('_', ' ')} ({st.definition()})"
        claim = PHRASE[kg.relation_labels[r]].format(h=nh, t=nt)
        return f"Lexical semantics question about WordNet word senses.\nClaim: {claim}.\n\nIs this claim correct? Answer yes or no."
    verdicts = judge(items, prompt)
    res = summarize(items, verdicts, extra_fields=("true_by_wordnet",))
    # among suggestions WordNet cannot confirm, how many do both judges accept?
    for s in ("RotatE top-1", "AnyBURL top-1"):
        unv = [it for it in items if it["source"] == s and it["verifier"] == "unverified"]
        if unv:
            res[s]["unverified_n"] = len(unv)
            res[s]["unverified_judged_true_by_majority"] = float(np.mean([sum(verdicts[j][it["key"]] for j in JUDGES) * 2 > len(JUDGES) for it in unv]))
    return {"summary": res}


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["GO"] = go_part(); OUT.write_text(json.dumps(res, indent=2)); log("GO", json.dumps(res["GO"]["summary"], indent=1))
    res["WN18RR"] = wn_part(); OUT.write_text(json.dumps(res, indent=2)); log("WN", json.dumps(res["WN18RR"]["summary"], indent=1))
    log("done")


if __name__ == "__main__":
    main()
