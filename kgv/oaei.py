"""OAEI Anatomy track: Adult Mouse Anatomy (MA) vs. the NCI Thesaurus anatomy fragment (human).

Two independently built ontologies, different naming conventions ("Upper_Extremity" vs "forelimb"),
with a curated reference alignment. This is the real version of "two graphs, different terms".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import rdflib
from rdflib.namespace import OWL, RDF, RDFS

DATA = Path(__file__).resolve().parents[1] / "data" / "oaei" / "anatomy"
OBO = rdflib.Namespace("http://www.geneontology.org/formats/oboInOwl#")


@dataclass
class Onto:
    name: str
    classes: list[str]
    label: dict[str, str]
    synonyms: dict[str, list[str]]
    edges: list[tuple[str, str, str]] = field(default_factory=list)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("_", " ").lower()).strip()


def load(name: str) -> Onto:
    g = rdflib.Graph()
    g.parse(DATA / f"{name}.owl", format="xml")
    classes = sorted(str(c) for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, rdflib.URIRef))
    cs = set(classes)
    label = {c: norm(str(g.value(rdflib.URIRef(c), RDFS.label) or c.split("#")[-1])) for c in classes}
    syn: dict[str, list[str]] = {}
    for c in classes:
        out = []
        for p in (OBO.hasRelatedSynonym, OBO.hasExactSynonym, OBO.hasSynonym):
            for node in g.objects(rdflib.URIRef(c), p):
                lab = g.value(node, RDFS.label) if not isinstance(node, rdflib.Literal) else node
                if lab:
                    out.append(norm(str(lab)))
        syn[c] = sorted(set(out) - {label[c]})
    edges = []
    for c in classes:
        for sup in g.objects(rdflib.URIRef(c), RDFS.subClassOf):
            if isinstance(sup, rdflib.URIRef) and str(sup) in cs:
                edges.append((c, "is_a", str(sup)))
            elif isinstance(sup, rdflib.BNode):
                prop = g.value(sup, OWL.onProperty)
                tgt = g.value(sup, OWL.someValuesFrom)
                if prop is not None and tgt is not None and str(tgt) in cs:
                    edges.append((c, str(prop).split("#")[-1].replace("UNDEFINED_", ""), str(tgt)))
    return Onto(name, classes, label, syn, sorted(set(edges)))


def reference(file: str = "reference.rdf") -> list[tuple[str, str]]:
    g = rdflib.Graph()
    g.parse(DATA / file, format="xml")
    A = rdflib.Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment")
    pairs = []
    for cell in g.subjects(RDF.type, A.Cell):
        pairs.append((str(g.value(cell, A.entity1)), str(g.value(cell, A.entity2))))
    if not pairs:  # files without rdf:type on cells
        e1 = list(g.subject_objects(A.entity1))
        e2 = dict(g.subject_objects(A.entity2))
        pairs = [(str(o), str(e2[s])) for s, o in e1 if s in e2]
    return sorted(set(pairs))
