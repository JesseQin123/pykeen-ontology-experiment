# Do graph embeddings actually help on semantic ontology graphs? (final report)

2026-10-17 · 17 experiments · PyKEEN 1.11 · HTML version with full tables: [docs/report.html](docs/report.html) ·
中文版：[REPORT.zh.md](REPORT.zh.md)

The question: with PyKEEN-style graph embeddings, (1) can we discover real relationships the graph is missing,
(2) do we get useful suggested connections, (3) does it help performance on large graphs, and (4) given two graphs that
use different terms for the same entities, will the embeddings find the correspondences and suggest mappings?
Each claim became a measurable experiment, and every experiment is paired with a non-embedding baseline. All numbers
come from `results/SUMMARY.json`, `results/STATS.json` and the per-experiment JSON files.

## Findings

| Claim | Verdict | Key evidence | Recommended practice |
|---|---|---|---|
| 1. Discovers missing relationships | True, but much weaker than it looks | GO trained on 2020, scored on 2026 curator additions: embedding + text fusion reaches Hits@10 0.30–0.36 on genuinely new links (0.37 on a 2018→2020 replication), rules 0.03–0.17. The same methods score MRR ≈ 0.5 on random hold-out but 0.10–0.19 prospectively. 58% of additions were already entailed; a reasoner ranks 93% of those in its top 10 | Reasoner first, then embedding + text fusion for candidates, then LLM and expert filtering |
| 2. Useful suggested connections | Usable only with filtering | Top-1 suggestions change 41% of the time across seeds; about 0.3% of suggestions for random concepts were adopted within six years; three LLM judges (kappa 0.75–0.81) accept about 9%; candidates + LLM majority vote enrich later-realized suggestions 89-fold and keep 87% of them | Fuse seeds and models; LLM vote as first filter; expert review |
| 3. Performance on large graphs | Faster queries, not better accuracy | 3-hop traversal p99 1.5 s at 3M nodes vs. HNSW 0.01 ms with 99.9% recall; under a fixed configuration RotatE MRR falls 0.62 → 0.15 as YAGO grows 5k → 123k entities (150 epochs: 0.16; lower learning rate: 0.21); rules 0.64 → 0.48 | Embeddings + ANN for approximate queries, graph database for exact ones; retune at each scale; place new concepts via a text-to-embedding mapping |
| 4. Naturally crosses taxonomies | Does not happen on its own | OAEI Anatomy (independently built): every seed-free structural method is at chance (Gromov-Wasserstein 0.07% Hits@1); with exact-label anchors, text + joint KGE lifts hard-pair Hits@1 from 83.4% to 88.6%; text + KGE + Claude verification F1 0.939 (AML, the winner: 0.941) | Exact-label anchors → text candidates → joint KGE correction → LLM verification |

## Experimental design

| Dataset | Purpose |
|---|---|
| WordNet (WN18RR) | Standard semantic-ontology benchmark; warm-up for Q1 and Q2 |
| Gene Ontology releases 2018 / 2020 / 2024 / 2026 | Train on an old release, test on a newer one: the only real test of discovery (Q1, Q2) |
| YAGO3-10 | 123k entities, 1.08M edges, for scale (Q3) |
| OAEI Anatomy (mouse vs. human) | Two independently built ontologies with different terms and an expert reference alignment (Q4) |
| FB15k-237 | A dense graph, as a control for density in alignment |

Baselines: popularity, the published AnyBURL rule learner, an is_a transitivity reasoner, label-text embeddings
(bge-large), and anchor-neighbourhood matching. LLMs: local qwen3.5-27B; Claude Opus 5.5, GPT-6.1 and Gemini 4 via a
relay. Protocol: filtered ranking, both directions, expected rank on ties; paired-bootstrap 95% CIs.

## Question 1: discovering missing relationships

**WordNet random hold-out (5,848 queries)**

| Method | MRR | Hits@1 | Hits@10 |
|---|---|---|---|
| Fusion + Claude re-ranking the top 20 | 0.580 | 0.519 | 0.651 |
| Fusion + GPT-6.1 re-ranking the top 20 | 0.568 | 0.499 | 0.648 |
| AnyBURL first, RotatE for the rest | 0.523 | 0.479 | 0.609 |
| Fusion (AnyBURL + RotatE + TransE) | 0.522 | 0.472 | 0.619 |
| AnyBURL | 0.517 | 0.475 | 0.592 |
| RotatE (3 seeds, sd 0.0014) | 0.487 | 0.434 | 0.587 |
| Our own path rules (length ≤ 2) | 0.424 | 0.405 | 0.459 |
| TransE | 0.214 | 0.012 | 0.541 |

AnyBURL − RotatE: MRR +0.029 [+0.023, +0.035]. The first report's "embeddings beat graph search" came from a rule
baseline that was too weak.

**GO 2020 → 2026 prospective test** (6,412 added links: 3,743 entailed, 2,669 genuinely new; 6,758 removed in the same period)

| Method (child → parent) | All MRR | All H@10 | New H@1 | New H@10 | Entailed H@10 |
|---|---|---|---|---|---|
| Reasoner (is_a transitivity) | 0.283 | 0.543 | 0.000 | 0.007 | 0.926 |
| Fusion (all methods + reasoner) | 0.260 | 0.473 | 0.069 | 0.280 | 0.611 |
| Fusion (RotatE + TransE + text) | 0.185 | 0.383 | 0.091 | 0.362 | 0.398 |
| RotatE + text | 0.142 | 0.298 | 0.081 | 0.299 | 0.297 |
| TransE | 0.113 | 0.344 | 0.012 | 0.273 | 0.394 |
| Poincaré (50-d, is_a only) | 0.104 | 0.218 | – | – | – |
| RotatE | 0.093 | 0.202 | 0.045 | 0.208 | 0.198 |
| AnyBURL | 0.092 | 0.177 | 0.048 | 0.173 | 0.180 |
| Text (bge-large) | 0.084 | 0.205 | 0.043 | 0.219 | 0.194 |
| Our own path rules (length ≤ 2) | 0.012 | 0.025 | 0.007 | 0.025 | 0.025 |

On new links, RotatE + text vs. RotatE: MRR +0.055 [+0.046, +0.063]; vs. text: +0.058 [+0.048, +0.069].
2018 → 2020 replication: RotatE + text reaches MRR 0.183 and H@10 0.366 on new links. Error detection (scoring links that
curators later removed lower than kept ones): AUC 0.65–0.69; rules 0.48.

**LLMs (Hits@1)**

| Task | Candidates alone | + qwen | + Claude | + GPT | Claude alone |
|---|---|---|---|---|---|
| GO new is_a links (all 2,386) | 8% | 18%* | 23.6% | 21.6% | 26%* |
| GO new concepts (all 1,465) | 29%* | 40%* | 54.8% | 51.7% | 58.7%* |
| OAEI hard pairs (580) | 83.8% | 94.3% | 96.5% | 95.5% | 82.2% |

\* 150-question sample. Contamination probe (naming the GO ID): Claude 75% for older concepts and 11% for concepts created
in 2024–26; GPT 46% / 0%.

## Question 2: suggested connections

- Seed stability: three RotatE runs differing only in seed agree on the top-1 suggestion 59% of the time and share 7 of the top 10 on average.
- Three-judge blind rating (Claude / GPT / Gemini, kappa 0.75–0.81). GO top suggestions judged correct by majority:
  RotatE + text 9%, TransE 9%, RotatE 7%, text 3%; controls: curator additions 68%, existing links 90%, random 0%.
  WordNet: AnyBURL 43%, RotatE 42%; among suggestions WordNet cannot confirm, 16% / 18.5% judged correct.
- Suggestion pipeline (1,000 concepts × 5): 0.3% came true within six years; 6.8% pass the LLM majority vote;
  3.8% of passed vs. 0.04% of rejected came true (89-fold); 87% of true ones kept.
- New-concept placement (1,465 concepts): text 16.0% / 40.6% (H@1 / H@10); + text → RotatE mapping 18.5% / 46.8%;
  + popularity prior 21.1% / 55.0%; + Claude re-ranking 54.8%.

## Question 3: large graphs

| Nodes | Brute-force kNN | 2-hop p99 | 3-hop p50 | 3-hop p99 |
|---|---|---|---|---|
| 10k | 0.07 ms | 0.8 ms | 0.6 ms | 1.5 ms |
| 1M | 6.6 ms | 124 ms | 43 ms | 270 ms |
| 3M | 19 ms | 704 ms | 268 ms | 1,535 ms |

HNSW over real RotatE embeddings: 0.009 ms per query, 99.9% recall. Training: about 48 s per million triples per epoch.

| YAGO (same 1,000 test triples) | Edges | Rules MRR | RotatE MRR |
|---|---|---|---|
| 4,874 entities | 15k | 0.639 | 0.619 |
| 14,930 | 56k | 0.578 | 0.522 |
| 39,979 | 208k | 0.526 | 0.344 |
| 123,143 | 1.08M | 0.478 | 0.147 |
| Full, 150 epochs | | | 0.156 |
| Full, learning rate 2e-3 | | | 0.21 (flat from 10 to 50 epochs) |

## Question 4: crossing taxonomies

Seed-free (Hits@1): FB15k-237 views GW 13.8%, WordNet views 1.9%, independently built OAEI ontologies 0.07% (chance 0.03%).

OAEI hard pairs (580): structure without anchors 0.2%; anchor-neighbourhood 7.6%; joint RotatE over automatic anchors
21.7%; string similarity 70.9%; bge-large 83.4%; bge-large + joint RotatE 88.6%; + Claude re-ranking 96.5%; Claude alone 82.2%.

| Full alignment | P | R | F1 |
|---|---|---|---|
| Exact label matches | 0.990 | 0.613 | 0.758 |
| + text + Claude verification | 0.965 | 0.902 | 0.932 |
| + text + graph embeddings + Claude verification | 0.964 | 0.916 | 0.939 |
| + text + graph embeddings + GPT verification | 0.975 | 0.902 | 0.937 |
| AML / Matcha (winners, literature) | 0.956 | 0.927 | 0.941 |
| OLaLa (LLM-based, literature) | 0.924 | 0.896 | 0.910 |
| LogMap (literature) | 0.917 | 0.848 | 0.881 |

## Model choice (WordNet, shared configuration)

RotatE 0.497, PairRE 0.483, CrossE 0.428, ComplEx 0.385, SimplE 0.376, TransE 0.223; TransD, DistMult, TransH, QuatE,
MuRE and AutoSF ≤ 0.06 (failed to train); TuckER and BoxE could not run on MPS.

## The converged method

1. **Completing an ontology:** reasoner → embedding (RotatE + TransE) + text fusion for candidates → AnyBURL for explainable paths → LLM vote → expert review.
2. **Suggested connections:** fuse seeds and models; an LLM vote cuts review volume to about 7%; record expert decisions to measure real precision.
3. **Large graphs:** embeddings + HNSW for approximate "related concept" queries, graph database for exact ones; retune at each scale; place new concepts with a text-to-embedding mapping.
4. **Cross-ontology mapping:** exact-label anchors → text candidates with synonyms → joint graph embedding as structural correction → LLM verification → (recommended) logical repair.
5. **Evaluation:** prefer prospective tests to random hold-out; report entailed and new links separately; always include a strong rule baseline, a reasoner and a text baseline; rate suggestions blind with positive and negative controls.

## What it takes to confirm this on the original ontology

1. The ontology (or an anonymized sample), the PyKEEN configuration, and the specific examples that looked useful.
2. Reproduce those examples with the rule, reasoner and text baselines to see whether embeddings were needed.
3. With version history, apply the GO prospective protocol (code ready).
4. 50 suggestions each from embeddings, rules and random, rated blind by experts and the three LLM judges.
5. With a second vocabulary, apply the OAEI four-step pipeline and have experts review 100–200 pairs.

## Limitations

Limited hyperparameter search; "came true in GO within six years" is a strict lower bound; LLMs have memorized public
ontologies; only the OAEI Anatomy track was used and no logical repair was done; traversal used single-threaded scipy;
Gemini's relay quota ran out, so the full WordNet re-ranking uses only GPT and Claude.

## Reproducing

```bash
uv sync
uv run python experiments/exp4_go_temporal.py
uv run python experiments/exp11_anyburl.py
uv run python experiments/exp5_oaei_anatomy.py
uv run python experiments/exp15_frontier_llm.py
uv run python experiments/exp16_llm_blind_rating.py
uv run python experiments/exp17_llm_on_structure.py
uv run python experiments/summarize.py && uv run python experiments/stats.py
```

Inference runs on hosted endpoints configured in a gitignored `.env` (`RELAY_BASE_URL`, `RELAY_API_KEY`,
`OPENROUTER_BASE_URL`, `OPENROUTER_API_KEY`); training runs locally (Apple MPS). AnyBURL needs Java
(`brew install openjdk`) and `tools/anyburl/AnyBURL-23-1x.jar`. GO releases and OAEI data download into `data/` (not committed).

| Experiment | File |
|---|---|
| 1 WN18RR link prediction and suggestion checks | `exp1_link_prediction.py` |
| 2 query latency, training cost, ANN | `exp2_scaling.py`, `exp2b_ann_realistic.py` |
| 3 alignment of renamed views | `exp3_alignment.py` |
| 4 GO prospective test | `exp4_go_temporal.py` |
| 5 OAEI Anatomy | `exp5_oaei_anatomy.py`, `exp5b_oaei_ablation.py` |
| 6 seed-free alignment (GW) | `exp6_unsupervised_alignment.py` |
| 7 model zoo | `exp7_model_zoo.py` |
| 8 YAGO scale and training curves | `exp8_scale_accuracy.py`, `exp8b_yago_training_curve.py` |
| 9 Poincaré | `exp9_hyperbolic_go.py` |
| 10 new-concept placement | `exp10_new_terms.py` |
| 11 AnyBURL | `exp11_anyburl.py` |
| 12 seed stability | `exp12_seed_stability.py` |
| 13 fusion | `exp13_ensembles.py` |
| 14 question sets, local LLM | `exp14_llm.py` |
| 15 frontier LLMs | `exp15_frontier_llm.py` |
| 16 three-judge blind rating | `exp16_llm_blind_rating.py` |
| 17 LLMs on top of structure | `exp17_llm_on_structure.py` |
