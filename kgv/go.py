"""Gene Ontology releases as a temporal knowledge graph.

Train on an old release, then ask which edges curators *added later* between terms that already existed.
That is a prospective test of "embeddings discover real relationships the graph is missing", with no
simulated hold-out: the answers were written by GO curators years after the training snapshot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import KG

RELATIONS = ("is_a", "part_of", "regulates", "positively_regulates", "negatively_regulates")
DATA = Path(__file__).resolve().parents[1] / "data" / "go"


@dataclass
class Term:
    id: str
    name: str = ""
    namespace: str = ""
    definition: str = ""
    obsolete: bool = False
    synonyms: tuple = ()


def parse(release: str):
    """Return ({id: Term}, set of (head, relation, tail)) for a go-basic.obo release."""
    terms: dict[str, Term] = {}
    edges: set[tuple[str, str, str]] = set()
    cur = None
    syn: list[str] = []
    for line in (DATA / f"go-basic-{release}.obo").read_text().splitlines():
        if line.startswith("["):
            if cur is not None:
                cur.synonyms = tuple(syn)
            cur, syn = (Term("") if line == "[Term]" else None), []
            continue
        if cur is None or ": " not in line:
            continue
        k, v = line.split(": ", 1)
        if k == "id":
            cur.id = v
            terms[v] = cur
        elif k == "name":
            cur.name = v
        elif k == "namespace":
            cur.namespace = v
        elif k == "def":
            m = re.match(r'"(.*)"', v)
            cur.definition = m.group(1) if m else v
        elif k == "is_obsolete" and v.strip() == "true":
            cur.obsolete = True
        elif k == "synonym":
            m = re.match(r'"(.*)" (EXACT|RELATED|NARROW|BROAD)', v)
            if m and m.group(2) == "EXACT":
                syn.append(m.group(1))
        elif k == "is_a":
            edges.add((cur.id, "is_a", v.split()[0]))
        elif k == "relationship":
            rel, tgt = v.split()[:2]
            if rel in RELATIONS:
                edges.add((cur.id, rel, tgt))
    if cur is not None:
        cur.synonyms = tuple(syn)
    live = {i for i, t in terms.items() if not t.obsolete}
    edges = {e for e in edges if e[0] in live and e[2] in live}
    return terms, edges


def closure_is_a(edges) -> dict[str, set[str]]:
    parents: dict[str, set[str]] = {}
    for h, r, t in edges:
        if r == "is_a":
            parents.setdefault(h, set()).add(t)
    memo: dict[str, set[str]] = {}

    def anc(x):
        if x in memo:
            return memo[x]
        memo[x] = set()
        out = set()
        for p in parents.get(x, ()):
            out.add(p)
            out |= anc(p)
        memo[x] = out
        return out

    for x in list(parents):
        anc(x)
    return memo


def temporal(old: str, new: str):
    """KG trained on `old`; test = edges present in `new` but not `old`, among terms live in both releases.

    Each test edge is tagged 'entailed' when the old release already implied it through is_a transitivity
    (a curator just made an implied link explicit) or 'novel' otherwise.
    """
    t_old, e_old = parse(old)
    t_new, e_new = parse(new)
    live_both = {i for i, t in t_old.items() if not t.obsolete and i in t_new and not t_new[i].obsolete}
    train = sorted(e for e in e_old if e[0] in live_both and e[2] in live_both) + sorted(
        e for e in e_old if not (e[0] in live_both and e[2] in live_both))
    added = sorted(e for e in e_new - e_old if e[0] in live_both and e[2] in live_both)
    removed = sorted(e for e in e_old - e_new if e[0] in live_both and e[2] in live_both)

    ents = sorted({x for h, _, t in e_old for x in (h, t)})
    added = [e for e in added if e[0] in set(ents) and e[2] in set(ents)]
    eid = {e: i for i, e in enumerate(ents)}
    rid = {r: i for i, r in enumerate(RELATIONS)}
    to_arr = lambda es: np.array([(eid[h], rid[r], eid[t]) for h, r, t in es], dtype=np.int64).reshape(-1, 3)

    anc = closure_is_a(e_old)
    def kind(e):
        h, r, t = e
        if r == "is_a" and t in anc.get(h, ()):
            return "entailed"
        if r != "is_a" and any((a, r, t) in e_old for a in anc.get(h, ())):
            return "entailed"  # relation inherited from an ancestor
        return "novel"

    from pykeen.triples import TriplesFactory
    tr = to_arr(train)
    tf = TriplesFactory(mapped_triples=__import__("torch").as_tensor(tr), entity_to_id=eid, relation_to_id=rid)
    kg = KG(name=f"GO {old}->{new}", num_entities=len(ents), num_relations=len(RELATIONS), train=tr,
            valid=np.zeros((0, 3), np.int64), test=to_arr(added), entity_labels=ents,
            relation_labels=list(RELATIONS), factory=tf)
    info = {
        "terms_old": sum(not t.obsolete for t in t_old.values()), "edges_old": len(e_old),
        "edges_added": len(added), "edges_removed": len(removed),
        "added_kinds": {k: sum(kind(e) == k for e in added) for k in ("entailed", "novel")},
        "added_by_relation": {r: sum(e[1] == r for e in added) for r in RELATIONS},
    }
    kinds = np.array([kind(e) for e in added])
    return kg, info, kinds, to_arr([e for e in removed if e[0] in eid and e[2] in eid]), t_old
