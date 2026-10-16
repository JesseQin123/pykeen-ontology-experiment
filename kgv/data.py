"""Dataset loading and shared bookkeeping (filter sets, inverse directions)."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("pykeen").setLevel(logging.ERROR)


@dataclass
class KG:
    name: str
    num_entities: int
    num_relations: int
    train: np.ndarray  # (m, 3) int64 h, r, t
    valid: np.ndarray
    test: np.ndarray
    entity_labels: list[str]
    relation_labels: list[str]
    factory: object = None  # pykeen TriplesFactory for training
    _known_tails: dict = field(default=None, repr=False)

    @property
    def num_directions(self) -> int:
        return 2 * self.num_relations

    def known_tails(self) -> dict[tuple[int, int], set[int]]:
        """(head, direction) -> all true tails in train+valid+test. direction r+R is the inverse of r."""
        if self._known_tails is None:
            d: dict[tuple[int, int], set[int]] = {}
            R = self.num_relations
            for h, r, t in np.concatenate([self.train, self.valid, self.test]):
                d.setdefault((int(h), int(r)), set()).add(int(t))
                d.setdefault((int(t), int(r) + R), set()).add(int(h))
            self._known_tails = d
        return self._known_tails


def load(name: str) -> KG:
    from pykeen import datasets

    ds = {"wn18rr": datasets.WN18RR, "fb15k237": datasets.FB15k237}[name.lower()]()
    tf = ds.training
    ent = [None] * tf.num_entities
    for k, v in tf.entity_to_id.items():
        ent[v] = k
    rel = [None] * tf.num_relations
    for k, v in tf.relation_to_id.items():
        rel[v] = k
    return KG(
        name=name,
        num_entities=tf.num_entities,
        num_relations=tf.num_relations,
        train=tf.mapped_triples.numpy().astype(np.int64),
        valid=ds.validation.mapped_triples.numpy().astype(np.int64),
        test=ds.testing.mapped_triples.numpy().astype(np.int64),
        entity_labels=ent,
        relation_labels=rel,
        factory=tf,
    )


def as_queries(triples: np.ndarray, num_relations: int) -> np.ndarray:
    """Turn triples into tail-prediction queries in both directions: (h, r, t) and (t, r+R, h)."""
    inv = np.stack([triples[:, 2], triples[:, 1] + num_relations, triples[:, 0]], axis=1)
    return np.concatenate([triples, inv])
