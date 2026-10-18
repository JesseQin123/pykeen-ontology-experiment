"""Experiment 4 - Prospective validation on the Gene Ontology: train on an old release, predict what curators added later.

  * test edges = edges present in the later release but not the old one, between terms that existed in both
  * 'entailed' = already implied by is_a transitivity in the old release; 'novel' = genuinely new knowledge
  * methods: popularity, path rules, RotatE (+ TransE), label-text similarity, structure+text hybrid,
    local-LLM re-ranking of the hybrid's top candidates
  * also: precision of top suggestions that later became true, and whether models flag edges curators later removed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgv import baselines, data, evaluate, go, kge  # noqa: E402

RESULTS, RUNS = ROOT / "results", ROOT / "runs"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class TextScorer:
    """Cosine similarity of term names (sentence embeddings). Relation-agnostic, plus a per-relation tail prior."""

    def __init__(self, kg, terms, model="BAAI/bge-small-en-v1.5", with_def=False):
        texts = [terms[e].name + ((": " + terms[e].definition) if with_def else "") for e in kg.entity_labels]
        from kgv import textenc
        if model in textenc.HOSTED:
            self.E = textenc.encode(texts, model)
        else:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(model, device="mps")
            self.E = st.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False).astype(np.float32)
        pop = baselines.Popularity(kg).pop
        self.prior = np.log1p(pop) * 1e-3

    def score(self, heads, dirs):
        s = self.E[heads] @ self.E.T + self.prior[dirs]
        s[np.arange(len(heads)), heads] = -1.0  # a term is trivially most similar to itself
        return s


class Reasoner:
    """Deductive baseline: is_a transitivity. Ancestors (or descendants) ranked by closeness in the hierarchy."""

    def __init__(self, kg):
        isa = kg.relation_labels.index("is_a")
        self.isa, self.R, self.n = isa, kg.num_relations, kg.num_entities
        self.up, self.down = {}, {}
        for h, r, t in kg.train:
            if r == isa:
                self.up.setdefault(int(h), []).append(int(t))
                self.down.setdefault(int(t), []).append(int(h))
        self.pop = np.log1p(np.bincount(kg.train[kg.train[:, 1] == isa][:, 2], minlength=self.n)) * 1e-4

    def _bfs(self, h, nbrs):
        dist, frontier, d = {}, [h], 0
        while frontier:
            d += 1
            nxt = []
            for x in frontier:
                for y in nbrs.get(x, ()):
                    if y not in dist and y != h:
                        dist[y] = d
                        nxt.append(y)
            frontier = nxt
        return dist

    def score(self, heads, dirs):
        out = np.tile(self.pop, (len(heads), 1))
        for i, (h, d) in enumerate(zip(heads, dirs)):
            if d % self.R != self.isa:
                continue
            for y, k in self._bfs(int(h), self.up if d < self.R else self.down).items():
                out[i, y] += 1.0 / k
        return out


def zfuse(*fns, weights=None):
    weights = weights or [1.0] * len(fns)

    def f(h, d):
        out = 0
        for w, fn in zip(weights, fns):
            s = np.asarray(fn(h, d), dtype=np.float64)
            out = out + w * (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
        return out
    return f


def split_metrics(r, kinds, rel, kg):
    q_kinds = np.concatenate([kinds, kinds])
    q_rel = np.concatenate([rel, rel])
    out = {"all": evaluate.summarize(r), "tail_only(child->parent)": evaluate.summarize(r[: len(kinds)])}
    for k in ("entailed", "novel"):
        out[k] = evaluate.summarize(r[q_kinds == k])
        out[k + "_tail_only"] = evaluate.summarize(r[: len(kinds)][kinds == k])
    for i, name in enumerate(kg.relation_labels):
        m = q_rel == i
        if m.any():
            out["rel:" + name] = evaluate.summarize(r[m])
    return out


def removal_auc(kg, removed, fn, n=4000, seed=0):
    """AUC that a later-removed edge scores lower than a random kept edge (higher = model flags errors)."""
    rng = np.random.default_rng(seed)
    rem = set(map(tuple, removed.tolist()))
    kept = np.array([t for t in kg.train[rng.permutation(len(kg.train))[: n * 2]] if tuple(t) not in rem])[:n]
    rem_s = removed[rng.permutation(len(removed))[:n]]

    def s(tr):
        out = []
        for i in range(0, len(tr), 256):
            b = tr[i : i + 256]
            sc = fn(b[:, 0], b[:, 1])
            out.append(sc[np.arange(len(b)), b[:, 2]])
        return np.concatenate(out)
    a, b = s(rem_s), s(kept)
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(b, a).statistic
    return float(u / (len(a) * len(b)))


def llm_rerank(kg, terms, kinds, fused, n=150, k=20, model="qwen3.5:27b", seed=0):
    """Ask a local LLM to choose the parent among the hybrid's top-k candidates for novel is_a additions."""
    import urllib.request
    rng = np.random.default_rng(seed)
    isa = kg.relation_labels.index("is_a")
    idx = [i for i in range(len(kg.test)) if kinds[i] == "novel" and kg.test[i, 1] == isa]
    idx = rng.permutation(idx)[:n]
    known = kg.known_tails()
    hits_base, hits_llm, in_cands, rows = 0, 0, 0, []
    for i in idx:
        h, r, t = map(int, kg.test[i])
        sc = fused(np.array([h]), np.array([r]))[0]
        sc[[x for x in known[(h, r)] if x != t]] = -np.inf
        sc[h] = -np.inf
        cands = list(np.argsort(-sc)[:k])
        hits_base += cands[0] == t
        if t not in cands:
            rows.append({"head": terms[kg.entity_labels[h]].name, "in_candidates": False})
            continue
        in_cands += 1
        opts = "\n".join(f"{j+1}. {terms[kg.entity_labels[c]].name}" for j, c in enumerate(cands))
        th = terms[kg.entity_labels[h]]
        prompt = (f"Gene Ontology term: \"{th.name}\"\nDefinition: {th.definition}\n\n"
                  f"Which ONE of these GO terms is the most appropriate direct parent (is_a) of it?\n{opts}\n\n"
                  "Answer with the number only.")
        body = json.dumps({"model": model, "prompt": prompt, "stream": False, "think": False,
                           "options": {"temperature": 0, "num_predict": 8}}).encode()
        try:
            req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            ans = json.loads(urllib.request.urlopen(req, timeout=600).read())["response"]
            pick = int("".join(ch for ch in ans if ch.isdigit())[:2] or 0) - 1
        except Exception as ex:  # noqa: BLE001
            pick, ans = -1, repr(ex)[:80]
        ok = 0 <= pick < len(cands) and cands[pick] == t
        hits_llm += ok
        rows.append({"head": th.name, "true": terms[kg.entity_labels[t]].name, "hybrid_top1": terms[kg.entity_labels[cands[0]]].name,
                     "llm_pick": terms[kg.entity_labels[cands[pick]]].name if 0 <= pick < len(cands) else ans, "llm_correct": bool(ok)})
    return {"n": len(idx), "k": k, "model": model, "true_parent_in_topk": in_cands / len(idx),
            "hybrid_Hits@1": hits_base / len(idx), "llm_rerank_Hits@1": hits_llm / len(idx), "examples": rows[:30]}


def realized_suggestions(kg, kinds, fns, terms, n_heads=3000, seed=0):
    """For random terms: is the top new is_a suggestion (not in / not implied by the old graph) added by curators later?"""
    rng = np.random.default_rng(seed)
    isa = kg.relation_labels.index("is_a")
    added = set(map(tuple, kg.test.tolist()))
    heads = rng.choice(kg.num_entities, n_heads, replace=False)
    old_anc = go.closure_is_a([(kg.entity_labels[h], kg.relation_labels[r], kg.entity_labels[t]) for h, r, t in kg.train])
    eid = {e: i for i, e in enumerate(kg.entity_labels)}
    known = kg.known_tails()
    out = {}
    for name, fn in fns.items():
        rows = []
        for s in range(0, n_heads, 256):
            b = heads[s : s + 256]
            sc = np.asarray(fn(b, np.full(len(b), isa)), dtype=np.float64)
            for j, h in enumerate(b):
                row = sc[j]
                excl = {x for x in known.get((int(h), isa), ()) if (int(h), isa, x) not in added}
                excl |= {eid[a] for a in old_anc.get(kg.entity_labels[h], ())}
                excl.add(int(h))
                row[list(excl)] = -np.inf
                c = int(np.argmax(row))
                rows.append((row[c], (int(h), isa, c) in added))
        rows.sort(key=lambda x: -x[0])
        out[name] = {f"top{p}%": float(np.mean([x[1] for x in rows[: max(1, len(rows) * p // 100)]])) for p in (1, 5, 10, 100)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="2020-01-01")
    ap.add_argument("--new", default="2026-08-05")
    ap.add_argument("--llm", default="qwen3.5:27b")
    ap.add_argument("--llm_n", type=int, default=150)
    args = ap.parse_args()
    tag = f"{args.old}_{args.new}"
    kg, info, kinds, removed, terms = go.temporal(args.old, args.new)
    log(info)
    res = {"setup": info, "methods": {}}
    rel = kg.test[:, 1]

    pop = baselines.Popularity(kg)
    rules = baselines.PathRules(kg)
    res["top_rules_is_a"] = rules.describe(6).get("is_a")
    fns = {"Popularity": pop.score, "PathRules": rules.score, "Reasoner(is_a transitivity)": Reasoner(kg).score}
    for name, seed, ov in (("RotatE", 0, dict(lr=1e-2)), ("TransE", 0, dict(lr=1e-3))):
        m, inf = kge.train(kg, name, seed=seed, device="mps", epochs=40 if name == "RotatE" else 100,
                           cache=RUNS / f"go_{tag}_{name}_s{seed}.pt", **ov)
        fns[name] = kge.score_fn(m, kg.num_relations)
        log(name, inf)
    text = TextScorer(kg, terms)
    fns["Text(bge-small name)"] = text.score
    text_def = TextScorer(kg, terms, with_def=True)
    fns["Text(bge-small name+def)"] = text_def.score
    fns["Hybrid(RotatE+Text)"] = zfuse(fns["RotatE"], fns["Text(bge-small name)"])
    fns["Hybrid(RotatE+Text+Rules)"] = zfuse(fns["RotatE"], fns["Text(bge-small name)"], fns["PathRules"],
                                             weights=[1.0, 1.0, 0.5])

    ranks = {}
    for name, fn in fns.items():
        ranks[name] = evaluate.ranks(kg, fn, kg.test)
        res["methods"][name] = split_metrics(ranks[name], kinds, rel, kg)
        m = res["methods"][name]
        log(f"{name:28s} all MRR={m['all']['MRR']:.3f} H10={m['all']['Hits@10']:.3f} | novel MRR={m['novel']['MRR']:.3f} "
            f"H10={m['novel']['Hits@10']:.3f} | entailed MRR={m['entailed']['MRR']:.3f}")
        (RESULTS / f"exp4_go_{tag}.json").write_text(json.dumps(res, indent=2))

    res["removed_edge_AUC(model flags later-removed edges)"] = {
        k: removal_auc(kg, removed, fns[k]) for k in ("RotatE", "TransE", "PathRules", "Text(bge-small name)", "Hybrid(RotatE+Text)")}
    log("removal AUC", res["removed_edge_AUC(model flags later-removed edges)"])
    res["realized_suggestions_is_a"] = realized_suggestions(
        kg, kinds, {k: fns[k] for k in ("PathRules", "RotatE", "Text(bge-small name)", "Hybrid(RotatE+Text)")}, terms)
    log("realized", res["realized_suggestions_is_a"])
    (RESULTS / f"exp4_go_{tag}.json").write_text(json.dumps(res, indent=2))
    if args.llm_n > 0:
        res["llm_rerank_novel_is_a"] = llm_rerank(kg, terms, kinds, fns["Hybrid(RotatE+Text)"], n=args.llm_n, model=args.llm)
        log("llm", {k: v for k, v in res["llm_rerank_novel_is_a"].items() if k != "examples"})
    np.savez_compressed(RESULTS / f"exp4_ranks_{tag}.npz", kinds=kinds, **{k.replace("/", "_"): v for k, v in ranks.items()})
    (RESULTS / f"exp4_go_{tag}.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
