"""Check predicted WN18RR triples against the full WordNet 3.0 database (open-world verification).

WN18RR is a 40k-synset slice of WordNet with a few relation types and only direct edges. A prediction that is
missing from WN18RR can still be *true*: either stored in full WordNet, or entailed by it (e.g. a hypernym of a
hypernym). That is exactly the "graph didn't capture it, embedding found it" claim, made testable.
"""
from __future__ import annotations

from functools import lru_cache

from nltk.corpus import wordnet as wn

from .data import KG


@lru_cache(maxsize=None)
def candidates(offset: str) -> tuple:
    """WordNet offsets are per-POS byte offsets, so one id can name a noun *and* a verb synset."""
    out = []
    for pos in "nvar":
        try:
            s = wn.synset_from_pos_and_offset(pos, int(offset))
        except Exception:
            s = None
        if s is not None and s.offset() == int(offset):
            out.append(s)
    return tuple(out)


_RESOLVED: dict[str, object] = {}


def synset(offset: str):
    if offset in _RESOLVED:
        return _RESOLVED[offset]
    c = candidates(offset)
    return c[0] if c else None


def resolve(kg) -> None:
    """Pick each entity's POS by voting: the candidate synset under which most of its training edges hold."""
    from collections import Counter

    votes: dict[int, Counter] = {}
    for h, r, t in kg.train:
        ch, ct = candidates(kg.entity_labels[h]), candidates(kg.entity_labels[t])
        if len(ch) < 2 and len(ct) < 2:
            continue
        accs = CANDIDATES[kg.relation_labels[r]]
        for sh in ch:
            for st in ct:
                if any(st in neighbours(sh, a) or sh in neighbours(st, a) for a in accs):
                    votes.setdefault(int(h), Counter())[sh] += 1
                    votes.setdefault(int(t), Counter())[st] += 1
    for e, label in enumerate(kg.entity_labels):
        c = candidates(label)
        if len(c) > 1 and e in votes:
            _RESOLVED[label] = votes[e].most_common(1)[0][0]


def _lemma_rel(s, attr):
    return {o.synset() for l in s.lemmas() for o in getattr(l, attr)()}


# candidate WordNet accessors for each WN18RR relation; the right one (and direction) is chosen empirically
CANDIDATES = {
    "_hypernym": ["hypernyms", "hyponyms"],
    "_instance_hypernym": ["instance_hypernyms", "instance_hyponyms"],
    "_member_meronym": ["member_meronyms", "member_holonyms"],
    "_has_part": ["part_meronyms", "part_holonyms"],
    "_synset_domain_topic_of": ["topic_domains", "in_topic_domains"],
    "_member_of_domain_region": ["region_domains", "in_region_domains"],
    "_member_of_domain_usage": ["usage_domains", "in_usage_domains"],
    "_also_see": ["also_sees"],
    "_verb_group": ["verb_groups"],
    "_similar_to": ["similar_tos"],
    "_derivationally_related_form": ["lemma:derivationally_related_forms"],
}


def neighbours(s, accessor: str) -> set:
    if accessor.startswith("lemma:"):
        return _lemma_rel(s, accessor.split(":", 1)[1])
    return set(getattr(s, accessor)())


class Verifier:
    def __init__(self, kg: KG):
        self.kg = kg
        resolve(kg)
        self.accessor: dict[int, str] = {}
        self.agreement: dict[str, float] = {}
        for r, name in enumerate(kg.relation_labels):
            tr = kg.train[kg.train[:, 1] == r][:2000]
            best, best_acc = None, -1.0
            for acc in CANDIDATES[name]:
                ok = sum(1 for h, _, t in tr if self.direct(h, acc, t))
                if ok / max(len(tr), 1) > best_acc:
                    best, best_acc = acc, ok / max(len(tr), 1)
            self.accessor[r] = best
            self.agreement[name] = best_acc

    def direct(self, h, acc, t) -> bool:
        sh, st = synset(self.kg.entity_labels[h]), synset(self.kg.entity_labels[t])
        return sh is not None and st is not None and st in neighbours(sh, acc)

    def verdict(self, h: int, r: int, t: int) -> str:
        """'direct' (stored in WordNet), 'entailed' (follows by transitivity / taxonomy), or 'unverified'."""
        acc = self.accessor[r]
        sh, st = synset(self.kg.entity_labels[h]), synset(self.kg.entity_labels[t])
        if sh is None or st is None:
            return "unverified"
        if st in neighbours(sh, acc):
            return "direct"
        name = self.kg.relation_labels[r]
        if name in ("_hypernym", "_instance_hypernym"):
            ancestors = set(sh.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))
            if st in ancestors:
                return "entailed"
        if name in ("_has_part", "_member_meronym"):
            # part-of inherited down the taxonomy: a part of my hypernym is (usually) a part of me
            for anc in sh.closure(lambda s: s.hypernyms() + s.instance_hypernyms()):
                if st in neighbours(anc, acc):
                    return "entailed"
        if name in ("_synset_domain_topic_of", "_member_of_domain_region", "_member_of_domain_usage"):
            for anc in sh.closure(lambda s: s.hypernyms()):
                if st in neighbours(anc, acc):
                    return "entailed"
        if name in ("_also_see", "_verb_group", "_similar_to", "_derivationally_related_form"):
            if sh in neighbours(st, acc):
                return "direct"  # symmetric relation stored in the other direction
        return "unverified"

    def label(self, e: int) -> str:
        s = synset(self.kg.entity_labels[e])
        return s.name() if s is not None else self.kg.entity_labels[e]
