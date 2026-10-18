"""Run the real AnyBURL rule learner (Meilicke et al.) and score its predictions with our shared protocol.

AnyBURL writes the top-k candidates (with confidences) for each test query. Candidates outside its top-k get the
realistic tie rank among the remaining entities, exactly as our evaluator treats tied scores.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .data import KG

ROOT = Path(__file__).resolve().parents[1]
JAR = ROOT / "tools" / "anyburl" / "AnyBURL-23-1x.jar"
JAVA = "/opt/homebrew/opt/openjdk/bin/java"


def _write(path: Path, triples, kg: KG):
    with open(path, "w") as f:
        for h, r, t in triples:
            f.write(f"e{h}\tr{r}\te{t}\n")


def run(kg: KG, workdir: Path, test: np.ndarray | None = None, seconds: int = 1000, threads: int = 12,
        top_k: int = 100, mem: str = "24G") -> dict:
    test = kg.test if test is None else test
    workdir.mkdir(parents=True, exist_ok=True)
    _write(workdir / "train.txt", kg.train, kg)
    _write(workdir / "valid.txt", kg.valid if len(kg.valid) else kg.train[:1], kg)
    _write(workdir / "test.txt", test, kg)
    rules = workdir / "rules"
    if not (workdir / f"rules-{seconds}").exists():
        (workdir / "learn.properties").write_text(
            f"PATH_TRAINING = {workdir/'train.txt'}\nPATH_OUTPUT = {rules}\nSNAPSHOTS_AT = {seconds}\nWORKER_THREADS = {threads}\n")
        subprocess.run([JAVA, f"-Xmx{mem}", "-cp", str(JAR), "de.unima.ki.anyburl.Learn", str(workdir / "learn.properties")],
                       check=False, cwd=workdir, stdout=open(workdir / "learn.log", "w"), stderr=subprocess.STDOUT)
        assert (workdir / f"rules-{seconds}").exists(), f"AnyBURL learn failed, see {workdir/'learn.log'}"
    preds = workdir / f"preds-{seconds}"
    (workdir / "apply.properties").write_text(
        f"PATH_TRAINING = {workdir/'train.txt'}\nPATH_VALID = {workdir/'valid.txt'}\nPATH_TEST = {workdir/'test.txt'}\n"
        f"PATH_RULES = {workdir}/rules-{seconds}\nPATH_OUTPUT = {preds}\nWORKER_THREADS = {threads}\nTOP_K_OUTPUT = {top_k}\n")
    subprocess.run([JAVA, f"-Xmx{mem}", "-cp", str(JAR), "de.unima.ki.anyburl.Apply", str(workdir / "apply.properties")],
                   check=False, cwd=workdir, stdout=open(workdir / "apply.log", "w"), stderr=subprocess.STDOUT)
    assert preds.exists(), f"AnyBURL apply failed, see {workdir/'apply.log'}"  # AnyBURL exits with status 1 on success
    return parse(preds)


def parse(path: Path) -> dict:
    """{(h, r, t): (head_candidates, tail_candidates)} with candidates as [(entity, conf)]."""
    out = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        h, r, t = lines[i].split()
        cand = lambda line: line.split(":", 1)[1].strip().split("\t") if line.split(":", 1)[1].strip() else []
        heads, tails = cand(lines[i + 1]), cand(lines[i + 2])  # "Heads: e47\t0.91\te9535\t0.06..."
        pair = lambda xs: [(int(xs[j][1:]), float(xs[j + 1])) for j in range(0, len(xs) - 1, 2)]
        out[(int(h[1:]), int(r[1:]), int(t[1:]))] = (pair(heads), pair(tails))
        i += 3
    return out


def ranks(kg: KG, preds: dict, test: np.ndarray | None = None) -> np.ndarray:
    """Filtered realistic ranks in the same order as evaluate.ranks: [tail queries..., head queries...]."""
    test = kg.test if test is None else test
    known = kg.known_tails()
    R, n = kg.num_relations, kg.num_entities
    tail_r, head_r = [], []
    for h, r, t in test:
        h, r, t = int(h), int(r), int(t)
        hc, tc = preds.get((h, r, t), ([], []))
        for (q, d, gold, cands, store) in ((h, r, t, tc, tail_r), (t, r + R, h, hc, head_r)):
            filt = known[(q, d)] - {gold}
            cands = [(e, c) for e, c in cands if e not in filt]
            confs = {e: c for e, c in cands}
            if gold in confs:
                g = confs[gold]
                greater = sum(c > g for c in confs.values())
                equal = sum(c == g for c in confs.values()) - 1
                store.append(greater + 1 + equal / 2)
            else:
                rest = n - len(filt) - len(cands)
                store.append(len(cands) + (rest + 1) / 2)
    return np.array(tail_r + head_r, dtype=float)
