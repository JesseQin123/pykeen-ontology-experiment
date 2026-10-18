# Literature survey: what knowledge-graph embeddings can and cannot do on semantic ontology graphs

*Compiled 2026-10-16 for the PyKEEN validation report. Scope: evidence for and against four claims about KGE (TransE, RotatE, ComplEx and related models, as in PyKEEN) applied to ontology-style graphs.*

## The four claims under test

1. KGE **finds real relationships** the graph is missing.
2. KGE gives **useful suggested connections**.
3. KGE **helps on large graphs**.
4. KGE **"naturally crosses semantic taxonomies"**: given two graphs that use different terms for the same entities, embeddings will find the similarities and suggest mappings.

## Our own results so far, for context

- On WN18RR/FB15k-237, RotatE reached MRR 0.487, against 0.424 for our path-rule baseline.
- Independently trained embeddings aligned with zero anchors scored at chance.
- Alignment only worked once we added anchors plus text embeddings.

## How the evidence was gathered

- Every paper below was located and checked during this survey: on arXiv, the ACL Anthology, the publisher's site, or the official benchmark or leaderboard page.
- Most numbers come from the papers' own result tables. We downloaded the PDFs and extracted the text.
- Some details could not be confirmed (an exact venue, or a number we could not open). These are marked **UNVERIFIED**.
- "MRR" means filtered MRR against all entities unless stated otherwise. Several large-scale results use **sampled** negatives, which inflates MRR. Each such case is flagged.

---

## A. How KGE is evaluated, and how it compares with rule-based and path-based baselines

### A.1 Re-evaluation studies: much of the reported progress came from evaluation and training artefacts

**Sun, Vashishth, Sanyal, Talukdar, Yang. "A Re-evaluation of Knowledge Graph Completion Methods." ACL 2020.** https://aclanthology.org/2020.acl-main.489/ (arXiv:1911.03903)
- Several neural KGC models gave many candidate triples the *same* score. They were evaluated with a "TOP" tie-breaking protocol that ranks the true answer first among ties, which inflated their results.
- Under an unbiased RANDOM tie-breaking protocol, FB15k-237 MRR fell sharply:
  - ConvKB: 0.396 → 0.243.
  - CapsE: 0.523 → 0.150.
  - KBAT: 0.518 → 0.157. KBAT also had test-data leakage in its original code.
- Sound evaluation alone moved these models from "state of the art" to below TransE-era baselines.

**Ruffinelli, Broscheit, Gemulla. "You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings." ICLR 2020.** https://iclr.cc/virtual_2020/poster_BkxSmlBFvr.html (results table: https://github.com/uma-pi1/kge)
- When old models get modern training (loss, regularisation, negative sampling, hyperparameter search), the gaps between architectures shrink or reverse.
- Filtered MRR on FB15k-237: RESCAL (2011) 0.356, ComplEx 0.348, RotatE 0.333, TransE 0.313.
- Filtered MRR on WN18RR: ComplEx 0.475, RotatE 0.478, RESCAL 0.467, TransE 0.228.
- A small random hyperparameter search was enough to find strong configurations.

**Rossi, Firmani, Matinata, Merialdo, Barbosa. "Knowledge Graph Embedding for Link Prediction: A Comparative Analysis." ACM TKDD 15(2), 2021.** https://arxiv.org/abs/2002.00819
- Compares 16 KGE models against one rule-based baseline, **AnyBURL**.
- AnyBURL against RotatE, filtered MRR:
  - FB15k-237: 0.324 vs 0.336.
  - WN18RR: 0.485 vs 0.475.
  - YAGO3-10: 0.528 vs 0.498.
  - FB15k: 0.835 vs 0.791 (H@1 81.1 vs 73.9).
- The authors conclude that AnyBURL "consistently ranks among the best models across almost all datasets and metrics". It trains in minutes to about 3 hours.
- Their structural analysis:
  - Predictions are easier when the query has many "source peers" and short or medium relational paths.
  - Predictions are hardest when there are many "target peers" and little path support.
  - Inverse or duplicate relations inflate results.

**Ali, Berrendorf, Hoyt, Vermue, Galkin, Sharifzadeh, Fischer, Tresp, Lehmann. "Bringing Light Into the Dark: A Large-scale Evaluation of Knowledge Graph Embedding Models Under a Unified Framework." IEEE TPAMI 2021/2022.** https://arxiv.org/abs/2006.13365
- The PyKEEN team re-implemented 21 models and ran thousands of experiments, totalling 24,804 GPU hours.
- Few published results reproduced "softly":
  - FB15K: 4 of 13.
  - WN18: 5 of 13.
  - FB15K-237: 2 of 3.
  - WN18RR: 3 of 5.
- "There is no single best configuration that works best for all dataset."
- With the right loss, training approach and inverse relations, TransE competes with state-of-the-art models on WN18RR.
- The authors recommend reporting the **realistic** (average) rank, because optimistic ranking distorts conclusions.

**Kadlec, Bajgar, Kleindienst. "Knowledge Base Completion: Baselines Strike Back." arXiv 2017.** https://arxiv.org/abs/1705.10744
- A well-tuned DistMult beat almost all models published on FB15k up to 2017.
- Gains claimed for new architectures were confounded with hyperparameter tuning. Peer-reviewed venue not confirmed; the paper is usually cited as an ACL RepL4NLP 2017 workshop paper (**UNVERIFIED**).

**Toutanova, Chen. "Observed versus latent features for knowledge base and text inference." CVSC workshop 2015.** https://aclanthology.org/W15-4007/
- A simple observed-feature (rule-like) model beat latent-feature models on FB15k and WN18, because those datasets are redundant through inverse relations.
- This finding led to FB15k-237.

**Akrami, Saeef, Zhang, Hu, Li. "Realistic Re-evaluation of Knowledge Graph Completion Methods: An Experimental Study." SIGMOD 2020.** https://doi.org/10.1145/3318464.3380599 (arXiv:2003.08001)
- Most triples in FB15k/WN18 and similar datasets belong to reverse or duplicate relations, or to Cartesian-product relations.
- Once that redundancy is removed, accuracy is substantially lower than reported.

### A.2 Strong rule-based and path-based baselines

**Meilicke et al. AnyBURL: "Anytime Bottom-Up Rule Learning for Knowledge Graph Completion." IJCAI 2019.** https://www.ijcai.org/proceedings/2019/435
**Journal extension: "Anytime bottom-up rule learning for large-scale knowledge graph completion." VLDB Journal 33(1):131–161, 2024.** https://doi.org/10.1007/s00778-023-00800-5
- AnyBURL learns cyclic and acyclic path rules bottom-up.
- It is competitive with latent (embedding) models and gives a rule-based explanation for every prediction.

**Ott, Meilicke, Samwald. "SAFRAN: An interpretable, rule-based link prediction method outperforming embedding models." AKBC 2021.** https://arxiv.org/abs/2109.08002
- SAFRAN adds a "non-redundant noisy-OR" way of combining AnyBURL rules.
- Its own Table 7 (average tie policy):

| Dataset | SAFRAN MRR | Strongest embeddings in the same table |
|---|---|---|
| FB15k-237 | **0.389** (H@1 0.298) | RESCAL 0.357, TuckER 0.352, ComplEx 0.348, RotatE 0.336 |
| WN18RR | **0.502** (H@1 0.459) | HAKE 0.497, ComplEx 0.477, RotatE 0.475 |
| YAGO3-10 | 0.564 | ComplEx 0.576 (the only one higher) |

- SAFRAN therefore beats most KGE models on FB15k-237 and WN18RR, while staying fully interpretable. Plain AnyBURL scores 0.355 / 0.494 / 0.559.
- Rule application takes seconds to minutes (WN18RR: 18.6 s).

**Zhu, Zhang, Xhonneux, Tang. "Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction." NeurIPS 2021.** https://arxiv.org/abs/2106.06935
- NBFNet is a path-based GNN that generalises path-ranking and rule methods.
- Filtered MRR:
  - FB15k-237: **0.415** vs RotatE 0.338 and DRUM (a rule learner) 0.343.
  - WN18RR: **0.551** vs RotatE 0.476 and DRUM 0.486.
- It uses about 3M parameters, against about 30M for TransE on FB15k-237.
- This is the clearest evidence that **path-based reasoning, not shallow embeddings, holds the state of the art among structure-only methods** on these benchmarks.

**What this means for our 0.487 vs 0.424 result.**
- Our path-rule baseline (0.424 on WN18RR) is far weaker than published rule systems (AnyBURL 0.485–0.494, SAFRAN 0.502) and path GNNs (NBFNet 0.551).
- A tuned rule learner would most likely match or beat RotatE's 0.487 on WN18RR.
- So "KGE beats rules" does not hold up once the rule baseline is strong.

### A.3 Realistic evaluation, open-world assumption, false negatives, stability

**Safavi, Koutra. "CoDEx: A Comprehensive Knowledge Graph Completion Benchmark." EMNLP 2020.** https://arxiv.org/abs/2009.07810
- **Frequency-only, non-learning baseline:**
  - On FB15k-237 it reaches MRR **0.236**, against 0.356 for the best tuned embedding (RESCAL).
  - The embedding beats it by under 5 MRR points for about 40% of FB15k-237 test triples, and by zero or less for 15%.
  - 15.98% of FB15k-237 test triples use relations skewed 50% or more toward a single head or tail. Example: `/people/person/gender` has 78.41% "male" tails.
- **Hard negatives:**
  - CoDEx took embedding models' top-10 predictions that were absent from the graph and labelled them by hand.
  - Many such predictions are plausible but *false*. Many others are true but missing.
  - Annotator agreement was 89.5%.
- This is direct evidence that "high-ranked but missing" does not mean "true".

**Yang, Lin, Zhang. "Rethinking Knowledge Graph Evaluation Under the Open-World Assumption." NeurIPS 2022.** https://arxiv.org/abs/2209.08858
- Under the open-world assumption, unknown triples include many true missing facts.
- MRR and Hits@K then grow only logarithmically with true model quality. Real improvements look insignificant, and **a stronger model can score lower**.
- Rankings between models become unreliable.
- They propose less top-heavy metrics (log-MRR, p-MRR).

**Cao, Ji, Lv, Li, Wen, Zhang. "Are Missing Links Predictable? An Inferential Benchmark for Knowledge Graph Completion." ACL 2021.** https://aclanthology.org/2021.acl-long.534
- InferWiki uses rule-guided splits, so every test triple has supporting evidence in training.
- It has manually annotated negative *and* unknown triples.
- Many links in FB15k-237 are not predictable from the available information at all. SimKGC (ACL 2022) cites this to explain its own results.

**Hoyt, Berrendorf, Galkin, Tresp, Gyori. "A Unified Framework for Rank-based Evaluation Metrics for Link Prediction in Knowledge Graphs." GLB workshop @ WWW 2022.** https://arxiv.org/abs/2203.07544
- Argues for chance-adjusted metrics (adjusted mean rank index, adjusted MRR), so that results can be compared across graphs of different sizes and against random baselines. This is directly applicable to PyKEEN reporting.

**Méroué, Gandon, Monnin. "Link Prediction or Perdition: the Seeds of Instability in Knowledge Graph Embeddings." ESWC 2026.** https://arxiv.org/abs/2606.03365
- This paper is the most relevant one for claim 2.
- Two RotatE runs that differ *only in random seed* share on average **4.3 of their top-10 predictions on FB15k-237, and 2.3 on WN18RR**.
- Top-1 overlap is at or below about 0.6 for most model/dataset pairs. Completing the graph with each run's top answer gives about **40% different triples** across runs.
- For RotatE on FB15k-237, about 25% of queries have **no** entity common to the top-10 of all five runs.
- Embedding neighbourhoods are unstable too: DistMult runs share 5.6 of 10 nearest neighbours.
- A better-MRR configuration is not guaranteed to be more stable. Voting ensembles help only a little.
- TransE was the exception, with top-1 overlap of 0.8–1.0.

**Bonner, Barrett, Ye, Swiers, Engkvist, Hoyt, Hamilton. "Understanding the performance of knowledge graph embeddings in drug discovery." Artificial Intelligence in the Life Sciences, 2022.** https://arxiv.org/abs/2105.10488
- Five KGE models on two biomedical KGs (Hetionet and BioKG).
- Training setup, hyperparameters, random seed and data split all have significant effects, and **can change the ranking of models**.
- Performance on general link prediction does not tell you how well a model does on the specific gene–disease task.
- RotatE was often best, but TransE was competitive when tuned.

**Background implementations.**
- Sun, Deng, Nie, Tang. "RotatE." ICLR 2019. https://arxiv.org/abs/1902.10197
- Ali et al. "PyKEEN 1.0." JMLR 22(82), 2021. https://www.jmlr.org/papers/v22/20-825.html

---

## B. Embeddings for OWL/OBO ontologies, hierarchy and transitivity

### B.1 Ontology-specific embeddings, and how plain KGE does on ontology completion

**Chen, Hu, Jiménez-Ruiz, Holter, Antonyrajah, Horrocks. "OWL2Vec\*: Embedding of OWL Ontologies." Machine Learning 110(7):1813–1845, 2021.** https://arxiv.org/abs/2009.14654
- OWL2Vec\* uses random walks, lexical annotations and logical constructors.
- Results from its Table 4 (MRR):

| Task | TransE | Best plain KGE | Pre-trained word2vec (labels only) | OWL2Vec\* |
|---|---|---|---|---|
| Subsumption, FoodOn | 0.029 | DistMult 0.076 | 0.136 | 0.213 (H@1 0.143) |
| Subsumption, GO | 0.015 | TransR 0.048 | 0.123 | 0.170 (H@1 0.076) |
| Class membership, HeLis | 0.181 | — | 0.899 | 0.953 |

- Plain KGE models are an order of magnitude worse than lexical baselines on real ontology subsumption.
- Most of the gain comes from **text** (labels and definitions). Adding the lexical document raised MRR by 26.9% on HeLis and 18.8% on FoodOn.
- Even the best method gets the right superclass first only 8–14% of the time.

**Kulmanov, Liu-Wei, Yan, Hoehndorf. "EL Embeddings: Geometric Construction of Models for the Description Logic EL++." IJCAI 2019.** https://www.ijcai.org/proceedings/2019/845
- ELEm represents classes as n-balls whose geometry follows EL++ semantics.
- It beat semantic-similarity measures and standard KGE on protein–protein interaction prediction.

**Jackermeier, Chen, Horrocks. "Dual Box Embeddings for the Description Logic EL++" (Box²EL). WWW 2024.** https://arxiv.org/abs/2301.11118
- Box²EL is a sound geometric EL++ embedding, compared against ELEm, EmEL++, BoxEL and ELBE.
- **Subsumption prediction** (predicting held-out stated axioms):

| Ontology | Box²EL MRR | H@1 | H@10 | Other baselines |
|---|---|---|---|---|
| GALEN | 0.10 | 0.05 | 0.20 | MRR 0.01–0.06 |
| GO | 0.10 | 0.04 | 0.23 (H@100 0.59) | — |
| Anatomy | 0.26 | 0.16 | 0.47 | — |

- **Approximating deductive reasoning** (ranking subsumptions that are *logically entailed* by the ontology but not stated):

| Ontology | H@1 | H@10 | MRR |
|---|---|---|---|
| GALEN | 0.01 | 0.09 | 0.03 |
| GO | 0.00 | 0.08 | 0.04 |
| Anatomy | 0.01 | 0.09 | 0.04 |

- In plain terms, the best purpose-built ontology embedding almost never ranks a subsumption that a reasoner derives in milliseconds first. The authors conclude that embeddings make "purely statistical predictions" and should be used *together with* a reasoner, not instead of one.

**Mashkova, Zhapa-Camacho, Hoehndorf. "Enhancing Geometric Ontology Embeddings for EL++ with Negative Sampling and Deductive Closure Filtering." NeSy 2024.** https://arxiv.org/abs/2405.04868 (journal extension: "DELE: Deductive EL++ Embeddings for Knowledge Base Completion", https://arxiv.org/abs/2411.01574)
- Standard negative sampling for EL++ embeddings treats *entailed* statements as negatives, and standard evaluation counts entailed statements as "novel" predictions.
- Both distort results. The fix is deductive-closure-aware negatives and filtering.
- **This applies directly to our setup.** If test edges in an ontology graph are entailed (for example through `is_a` transitivity), a KGE that "predicts" them has not discovered anything new.

**Chen, Mashkova, Zhapa-Camacho, Hoehndorf, He, Horrocks. "Ontology Embedding: A Survey of Methods, Applications and Resources." IEEE TKDE 2025.** https://arxiv.org/abs/2406.10964
- Reviews more than 80 papers and groups methods as geometric, sequence-based or graph-propagation.
- Introduces the mOWL library.
- The best starting point for choosing ontology-aware methods over generic KGE.

**Chen, He, Geng, Jiménez-Ruiz, Dong, Horrocks. "Contextual Semantic Embeddings for Ontology Subsumption Prediction" (BERTSubs). World Wide Web 26(5), 2023.** https://arxiv.org/abs/2202.09791
- A fine-tuned BERT with class-context templates. On five real ontologies it "substantially outperforms" KGE-based, word-embedding and OWL-embedding baselines.

**He, Chen, Jiménez-Ruiz, Dong, Horrocks. "Language Model Analysis for Ontology Subsumption Inference" (OntoLAMA). Findings of ACL 2023.** https://arxiv.org/abs/2302.06761
- Probing benchmarks for atomic and complex subsumption.
- Language models know less about ontology subsumption than about general natural-language inference, but improve markedly with a few examples.

**He, Chen, Dong, Jiménez-Ruiz, Hadian, Horrocks. "DeepOnto: A Python Package for Ontology Engineering with Deep Learning." Semantic Web journal, 2024.** https://arxiv.org/abs/2307.03067
- Python access to the OWL API, a reasoner, verbalisation, and LM-based tools for ontology alignment and completion.
- The practical tool for subsumption-completion experiments on OBO/OWL ontologies.

**Li, Bailleux, Bouraoui, Schockaert. "Ontology Completion with Natural Language Inference and Concept Embeddings: An Analysis." arXiv 2403.17216, 2024** (venue reported as EMNLP 2024 Findings: **UNVERIFIED**). https://arxiv.org/abs/2403.17216
- Natural-language-inference methods and concept embeddings complement each other, and hybrids work best.
- LLMs find ontology completion "highly challenging even after fine-tuning".

**Sun et al. "Are Large Language Models a Good Replacement of Taxonomies?" (TaxoGlimpse). PVLDB 17(11), 2024.** https://vldb.org/pvldb/vol17/p2919-sun.pdf
- Eighteen LLMs on ten taxonomies.
- Even the best LLM is unsatisfactory on specialised taxonomies and on entities near the leaves. Explicit taxonomies remain necessary.

### B.2 Hyperbolic embeddings for hierarchies

**Nickel, Kiela. "Poincaré Embeddings for Learning Hierarchical Representations." NeurIPS 2017.** https://arxiv.org/abs/1705.08039
- Reconstructing the **transitive closure** of the WordNet noun hierarchy:

| Method | Dimensions | MAP |
|---|---|---|
| Poincaré | 5 | 0.823 |
| Translational (TransE-like) | 5 | 0.517 |
| Euclidean | 5 | 0.024 |
| Euclidean | 200 | 0.168 |

- On held-out closure edges, Poincaré scores MAP 0.825 at 5 dimensions, against 0.490 for Euclidean at 200.
- Caveat: the model is *trained on the transitive closure*. The hierarchy's transitivity is handed to it rather than discovered.

**Ganea, Bécigneul, Hofmann. "Hyperbolic Entailment Cones for Learning Hierarchical Embeddings." ICML 2018.** https://proceedings.mlr.press/v80/ganea18a.html
- Models a partial order as nested convex cones, so transitivity holds by construction.
- Improves on Poincaré embeddings for embedding a DAG and for generalising on it.

**Balažević, Allen, Hospedales. "Multi-relational Poincaré Graph Embeddings" (MuRP). NeurIPS 2019.** https://arxiv.org/abs/1905.09791
- WN18RR MRR 0.477 at d=40 vs 0.459 for its Euclidean analogue MuRE.
- At d=200: 0.481 vs 0.475.
- The hyperbolic advantage is mostly a **low-dimension** effect.

**Chami, Wolf, Juan, Sala, Ravi, Ré. "Low-Dimensional Hyperbolic Knowledge Graph Embeddings" (AttH/RotH). ACL 2020.** https://arxiv.org/abs/2005.00545
- At d=32 on WN18RR: RotH MRR 0.472 vs RotE 0.463 vs RotatE 0.387. In high dimensions AttH reaches 0.496.
- Per relation (WN18RR, H@10, RotE → RotH):
  - hypernym: 0.237 → 0.276.
  - member_meronym: 0.320 → 0.399.
  - has_part: 0.291 → 0.346.
- **Important for ontologies:** even with hyperbolic geometry, the taxonomic `hypernym` relation reaches only H@10 ≈ 0.28. Missing-parent prediction is one of the *weakest* relations, not a strength.
- The user's "HypE" could not be matched to a specific peer-reviewed hierarchy KGE paper (**UNVERIFIED** / not located).

### B.3 Do KGE models handle transitivity and hierarchy?

**Gutiérrez-Basulto, Schockaert. "From Knowledge Graph Embedding to Ontology Embedding? An Analysis of the Compatibility between Vector Space Representations and Rules." KR 2018.** https://arxiv.org/abs/1805.10461
- Treats relations as regions in vector space and analyses which rule classes each embedding family can represent.
- Bilinear and translational models such as DistMult, ComplEx and TransE cannot faithfully capture many rule types, including some hierarchical and transitive patterns. (This summary of the specific theorems comes from the paper's framing; the individual results were not re-extracted.)

**Song, Luo, Huang. "Rot-Pro: Modeling Transitivity by Projection in Knowledge Graph Embedding." NeurIPS 2021.** https://arxiv.org/abs/2110.14450
- States that transitivity, "a very common relation pattern", is "still not fully supported" by existing KGE models, including RotatE.
- Proposes projections to fix it.

**Synthesis for B.**
- Standard KGE (TransE, RotatE, ComplEx) has no built-in transitivity or partial-order semantics.
- On real ontologies (GO, FoodOn, GALEN) its subsumption prediction is very weak: MRR 0.015–0.08.
- Ontology-specific geometric models do better but still have low absolute accuracy.
- Lexical and LM methods dominate whenever labels are available.
- Hyperbolic models help the hierarchy mainly at low dimension.
- None of them replaces a DL reasoner for entailed facts.

---

## C. Temporal and prospective validation: do predictions come true later?

Few studies evaluate this properly. The ones that do give modest, but non-zero, hit rates.

**Paliwal et al. (BenevolentAI). "Preclinical validation of therapeutic targets predicted by tensor factorization on heterogeneous graphs" (Rosalind). Scientific Reports 10:18250, 2020.** https://doi.org/10.1038/s41598-020-74922-z
- This is the strongest prospective study found.
- **Setup:** a ComplEx-style tensor factorisation on a biomedical KG. Edges were dated by their earliest publication, the model was trained up to a cutoff year (1990–2015), and it was tested on edges that appeared later.
- **Time-sliced results:**
  - With a 2005 cutoff, **recall@200 was 23.9%** over all later-discovered disease–gene therapeutic links. The authors summarise this as "1 in 4 therapeutic relationships eventually proven true".
  - Time-bounded mAP peaked at only 3.54%.
  - Recall@200 fell by about 50% after the first five years past the cutoff.
- **Rheumatoid arthritis:** 8 of 73 later-validated targets were in the top 500 (12% recall@500).
- **Wet-lab test:** 55 triaged targets were tested in patient-derived cells, giving **5 final hits (9%)**, against about 1% in typical RA screens.
- **Caveats:**
  - Only 51% of edges could be dated, and 74% of the dates were correct.
  - Predictions were filtered by experts before the lab test.
- This is real but low-precision discovery in a curated domain, aided by expert curation.

**Himmelstein et al. "Systematic integration of biomedical knowledge prioritizes drugs for repurposing" (Project Rephetio / Hetionet). eLife 2017.** https://elifesciences.org/articles/26726
- A path-based (degree-weighted path count) logistic model, not a KGE.
- AUROC: 97.4% on training indications, **85.5%** on external DrugCentral indications, **70.0%** on ClinicalTrials.gov indications.
- Performance drops as the validation set moves from curated to real-world (trial) evidence.

**Zeng et al. "Repurpose Open Data to Discover Therapeutics for COVID-19 using Deep Learning" (CoV-KGE). J. Proteome Research 2020.** https://arxiv.org/abs/2005.10831
- KGE on 15M edges mined from 24M PubMed papers.
- 41 candidate drugs were checked against transcriptomic and proteomic data and ongoing trials.
- No precision figure is given for predictions that later held up. The authors explicitly do not recommend specific drugs.

**Huang, Chandak, …, Zitnik. "A foundation model for clinician-centered drug repurposing" (TxGNN). Nature Medicine 2024.** https://doi.org/10.1038/s41591-024-03233-x
- A GNN with metric learning that makes zero-shot predictions across 17,080 diseases.
- Relative gains of 49.2% for indications and 35.1% for contraindications over 8 baselines.
- Predictions "align well" with off-label prescriptions in a large health system. That is retrospective agreement, not a prospective trial.

**Pu, Beck, Verspoor. "Graph embedding-based link prediction for literature-based discovery in Alzheimer's Disease." J. Biomedical Informatics 145, 2023.** https://minerva-access.unimelb.edu.au/server/api/core/bitstreams/c7964824-c75b-49cf-8c70-1e747dec3219/content
- 20 time-sliced train/test pairs, studying how the length of the prediction window affects success. Exact numbers were not extracted (**UNVERIFIED**).
- Related time-sliced studies (an AD-repurposing preprint, arXiv:2202.08712; COVID repurposing, arXiv:2010.09600) found that TransE-family models recover some triples published after the cutoff.

**Hu et al. "Open Graph Benchmark: Datasets for Machine Learning on Graphs." NeurIPS 2020.** https://arxiv.org/abs/2005.00687
**OGB leaderboard:** https://ogb.stanford.edu/docs/leader_linkprop/
- **ogbl-wikikg2 is a time split:**
  - Training: the Wikidata snapshot of May 2015.
  - Validation and test: triples added by August and November 2015.
  - Scale: 2.5M entities, 17.1M edges, 535 relations.
- **Leaderboard, test MRR:**
  - TransE (500 dimensions) 0.426.
  - RotatE (250 dimensions) 0.433.
  - ComplEx (250 dimensions) 0.403.
  - The best ensembles or text-augmented models reach 0.73–0.74, with up to 2.2B parameters.
- **Caveat:** the MRR is computed against only **1,000 sampled negatives**, not all 2.5M entities, so it overstates real ranking quality.

**Hu, Fey, Ren, Nakata, Dong, Leskovec. "OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs" (WikiKG90M). NeurIPS 2021 Datasets & Benchmarks.** https://arxiv.org/abs/2103.09430
- Another Wikidata time split, with 87M entities.
- Text features were decisive:

| Model | Test MRR (candidate-set version) |
|---|---|
| TransE-Shallow (17.4B params) | 0.741 |
| TransE-Concat (shallow + RoBERTa text) | 0.855 |
| ComplEx-Shallow | 0.588 |
| ComplEx-Concat | 0.864 |
| KDD Cup winners | 0.97 |

- Most provided candidates were "easy negatives". In **WikiKG90Mv2**, ranked against all entities, the best baseline (ComplEx-Concat) dropped to **MRR 0.1833**.

**Wang, Zhao, Wei, Liu. "SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models." ACL 2022.** https://arxiv.org/abs/2203.02167
- Human evaluation on Wikidata5M found that **about 49% of "wrong" Hits@1 predictions were actually correct**:
  - 24% for tail queries (54% genuinely wrong, 22% unknown).
  - 74% for head queries.
- Most such hits are on 1-to-N and N-to-1 relations (`instance of`, `place of birth`).
- Evidence for claim 1, but the "discoveries" are concentrated in generic, high-fan-out relations.

**Dong, Chen, He, Horrocks. "Ontology Enrichment from Texts: A Biomedical Dataset for Concept Discovery and Placement." CIKM 2023.** https://arxiv.org/abs/2306.14704
- Uses SNOMED CT releases (2014 vs 2017) with MedMentions to test whether new concepts can be placed in the right spot in the hierarchy.
- Placement is "very challenging": the best edge bi-encoder reaches **P@1 4.5% and R@10 19.9%**.
- GPT-3.5 reranking did not help.
- Spotting mentions that are not yet in the ontology (out-of-KB F1) is only 15–30%.

**Zhou et al. "The CAFA challenge reports improved protein function prediction and new functional annotations for hundreds of genes through experimental screens" (CAFA3). Genome Biology 20:244, 2019.** https://doi.org/10.1186/s13059-019-1835-8
- The gold standard for *prospective* Gene Ontology annotation prediction: predictions are scored on annotations that accumulate *after* the submission deadline.
- Experiments guided by predictions produced new annotations for more than 1,000 genes.
- Year-over-year improvement was only modest (none for Cellular Component). Not KGE-specific.

**Synthesis for C.**
- Time-split studies show that graph-based predictions do contain future facts. Examples:
  - About 24% recall@200 for therapeutic targets.
  - About half of "errors" judged correct on Wikidata.
  - Wet-lab hit rates around 9% after expert triage.
- Precision is low, the effect fades with the prediction horizon, and the best results come from domain-specific pipelines with text or curation, not raw KGE.
- **No study was found that time-validates PyKEEN-style KGE on an OBO/OWL ontology** (for example, GO release N against N+1). That makes it a gap worth filling with our own experiment.

---

## D. Ontology matching and entity alignment: does structure alone align graphs?

### D.1 OAEI ontology matching: lexical systems dominate, and plain string equality is a strong baseline

**OAEI Anatomy track** (mouse anatomy vs NCI Thesaurus human anatomy). Official results pages:
- 2021: https://oaei.ontologymatching.org/2021/results/anatomy/index.html
- 2022: https://oaei.ontologymatching.org/2022/results/anatomy/index.html
- 2023: https://oaei.ontologymatching.org/2023/results/anatomy/index.html
- 2024: https://oaei.ontologymatching.org/2024/results/anatomy/index.html

| System (year) | Precision | F1 | Recall | Recall+ (non-trivial matches) |
|---|---|---|---|---|
| **StringEquiv baseline** (normalised label equality) | 0.997 | **0.766** | 0.622 | 0.000 |
| AML (2021) | 0.956 | 0.941 | 0.927 | 0.810 |
| Matcha (2023/2024) | 0.951 | 0.941 | 0.931 | 0.818–0.82 |
| LogMap (2021–2024) | 0.917 | 0.881 | 0.848 | 0.602 |
| LogMapBio (2024) | 0.888 | 0.898 | 0.908 | 0.757 |
| OLaLa, LLM-based (2023) | 0.924 | 0.910 | 0.896 | 0.726 |
| SORBETMtch (2023) | 0.923 | 0.909 | 0.895 | 0.724 |
| MDMapper (2024) | 0.926 | 0.903 | 0.881 | 0.703 |
| ALOD2Vec, embedding-based (2021) | 0.828 | 0.796 | 0.766 | 0.382 |
| **OTMapOnto**, optimal-transport embeddings (2021) | 0.646 | **0.720** | 0.811 | 0.515 |

- Exact label matching alone gets F1 0.766 at 99.7% precision.
- The best systems (AML, Matcha) add only about 0.18 F1, mostly through synonyms, lexical expansion, background knowledge, and logical repair with a reasoner.
- An optimal-transport embedding matcher scored *below* the string baseline.
- Only LogMap and LogMapBio (and AML in 2021) produce **coherent** alignments, meaning no unsatisfiable classes after merging the two ontologies.

**He, Chen, Antonyrajah, Horrocks. "BERTMap: A BERT-Based Ontology Alignment System." AAAI 2022.** https://doi.org/10.1609/aaai.v36i5.20510 (arXiv:2112.02682)
- Fine-tunes BERT on synonym pairs taken from the ontologies' own labels, then extends and logically repairs the mappings.
- FMA–SNOMED: +1.4 F1 points over AML and +4.2 over LogMap.
- FMA–NCI: below both, by about 2.5 points.

**He, Chen, Dong, Jiménez-Ruiz, Hadian, Horrocks. "Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching" (Bio-ML). ISWC 2022.** https://arxiv.org/abs/2205.03447
- Equivalence F1 with 90% of mappings held out for testing:

| Task | EditSim (string similarity only) | LogMap | AML | BERTMap |
|---|---|---|---|---|
| OMIM–ORDO | 0.620 | 0.622 | 0.607 | 0.641 |
| NCIT–DOID | 0.838 | 0.773 | 0.820 | 0.868 |
| SNOMED–FMA (Body) | **0.787** | 0.636 | 0.807 | 0.773 |
| SNOMED–NCIT (Neoplas) | **0.759** (best) | 0.657 | 0.636 | 0.711 |

- A plain edit-distance baseline is within a few points of, and sometimes ahead of, the state of the art. **The signal in real ontology matching is overwhelmingly lexical.**

**He, Chen, Dong, Horrocks. "Exploring Large Language Models for Ontology Alignment." ISWC 2023 Posters & Demos (CEUR).** https://arxiv.org/abs/2309.07172
- Hard Bio-ML subsets with string-matchable pairs removed.
- NCIT–DOID: zero-shot Flan-T5-XXL (with a threshold) F1 **0.721**, vs BERTMap 0.628.
- SNOMED–FMA: 0.346 vs BERTMap 0.552.
- GPT-3.5-turbo: F1 0.313 and 0.132.
- Adding parent/child context did *not* help.

**Hertling, Paulheim. "OLaLa: Ontology Matching with Large Language Models." K-CAP 2023.** https://arxiv.org/abs/2311.03837
- Sentence-embedding candidates, then open LLMs (such as Llama-2-70B) decide yes or no.
- Few-shot prompting gives results "en par with supervised matching systems". On OAEI 2023 Anatomy it ranked 2nd (F1 0.910).
- Very slow: 9,259 s on Anatomy, against 10 s for LogMap.

**Qiang, Wang, Taylor. "Agent-OM: Leveraging LLM Agents for Ontology Matching." PVLDB 18(3):516–529, 2025.** https://arxiv.org/abs/2312.00326
- LLM agents with retrieval and matching tools.
- "Very close to" the long-standing best on simple OAEI tasks, with significant gains on complex and few-shot tasks.

**Taboada, Martínez, Arideh, Mosquera. "Ontology matching with Large Language Models and prioritized depth-first search" (MILA). Information Fusion 123:103254, 2025.** https://arxiv.org/abs/2501.11441
- Retrieve, identify, then prompt the LLM only on borderline cases.
- Best F1 in 5 of 7 unsupervised tasks from OAEI 2024 (up to +17%). Domain systems such as LogMapBio stayed better on OMIM–ORDO.

**Lushnei, Shumskyi, Shykula, Jiménez-Ruiz, d'Avila Garcez. "Large Language Models as Oracles for Ontology Alignment." EACL 2026.** https://arxiv.org/abs/2508.08500
- The LLM validates only the uncertain mappings from LogMap.
- OAEI 2025 Bio-ML official results: LogMapLLM averages F1 0.751, against 0.718 for LogMap and 0.762 for LogMapBio (2nd overall).
- Precision rose from 0.826 to 0.903; recall was unchanged.
- **The 2025 state of the art is a lexical/logical matcher plus an LLM as a text-reading judge. It is not structural embedding.**

### D.2 Embedding-based entity alignment: benchmark numbers depend on seeds and names

**Original models:**
- Chen, Tian, Yang, Zaniolo. **MTransE**, IJCAI 2017. https://www.ijcai.org/proceedings/2017/209
- Wang, Lv, Lan, Zhang. **GCN-Align**, EMNLP 2018. https://aclanthology.org/D18-1032/
- Sun, Hu, Zhang, Qu. **BootEA**, IJCAI 2018. https://www.ijcai.org/proceedings/2018/611
- Wu et al. **RDGCN**, IJCAI 2019. https://arxiv.org/abs/1908.08210

All of these need **seed alignments** (pre-aligned entity pairs used as training data). RDGCN and the GCN-Align attribute variant also initialise entities from their (translated) **names**.

**Sun, Zhang, Hu, Wang, Chen, Akrami, Li. "A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs" (OpenEA). PVLDB 13(11), 2020.** https://arxiv.org/abs/2003.07743
- 12 methods, all using **20% of the gold alignment as seeds**.
- Hits@1 on EN-FR-15K (V1):
  - MTransE 0.247.
  - GCN-Align 0.338.
  - BootEA 0.507.
  - MultiKE 0.749 (uses names/literals).
  - RDGCN 0.755 (uses names).
- With **relation triples only** (no attributes or names), on EN-FR-15K:
  - RDGCN falls to **0.255**.
  - MultiKE falls to 0.337.
  - BootEA reaches 0.507, even with 20% seeds.
- **The classic unsupervised, name-based system PARIS (Suchanek et al., VLDB 2011; cited through OpenEA) beats all of them without any seeds:**
  - F1 0.903 on EN-FR-15K and 0.935 on EN-DE-15K.
  - The best embedding methods with seeds reach 0.755 and 0.830.
- LogMap fails completely on DBpedia–Wikidata, because Wikidata URIs carry no meaningful local names (for example, P69). The authors report that "the results of all approaches on D-W decrease severely."
- All current approaches "require seed alignment as supervision". Unsupervised entity alignment is listed as future work.

**Liu, Cao, Pan, Li, Liu, Chua. "Exploring and Evaluating Attributes, Values, and Structures for Entity Alignment" (AttrGNN). EMNLP 2020.** https://aclanthology.org/2020.emnlp-main.515
- Shows that DBP15K is **name-biased**.
- A *names-only* baseline, NameBERT (BERT embeddings of entity names), reaches Hits@1:
  - ZH-EN 60.4.
  - JA-EN 74.5.
  - FR-EN 87.4.
- It beats every structure-only method in that comparison. Examples: BootEA 62.9 / 62.2 / 65.3; GCN-Align 41.3 / 39.9 / 37.3; MTransE 30.8 / 27.9 / 24.4.
- On DBP–YAGO, 77.6% of gold pairs have *identical* names and NameBERT reaches about 100%.
- In a "hard" split built from pairs with dissimilar names, every model drops. Structure-only BootEA, for example, scores 51.3 / 49.3 / 51.3.

**Liu, Hong, Wang, Chen, Kharlamov, Dong, Tang. "SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs." WWW 2022.** https://arxiv.org/abs/2203.01044
- Uses no labelled seeds, but entities are initialised with **LaBSE multilingual name embeddings**.
- LaBSE name embeddings *before any graph training* already reach Hits@1:
  - FR-EN 0.899.
  - JA-EN 0.689.
  - ZH-EN 0.581.
  - DWY100K DBP–YAGO 1.000.
- After SelfKG's structural refinement: 0.957, 0.816 and 0.745.
- So "self-supervised" alignment here is *name embeddings plus structural refinement*, not alignment from structure alone.

**Tang, Zhao, Li. "A Fused Gromov-Wasserstein Framework for Unsupervised Knowledge Graph Entity Alignment" (FGWEA). Findings of ACL 2023.** https://aclanthology.org/2023.findings-acl.205 (arXiv:2305.06574)
- **This is the most direct answer to "does structure alone align graphs without seeds?"**
- Its ablation includes a **GW-only** variant: optimal-transport matching on graph structure only, with no seeds and no names.
- GW-only results:
  - DBP15K ZH-EN Hits@1 **0.011**.
  - SRPRS EN-FR 0.004.
  - F1 **0.1%** on D-W-15K and MED-BBK-9K.
- Name-embedding matching alone ("Emb-Match") gets 0.763 and 0.915.
- The full method uses names first, then structure, and reaches 0.987.
- The authors: "directly optimizing GWD between KG structures is ineffective."
- This matches our own finding that zero-anchor alignment of independently trained embeddings is at chance.

**Xu, Luo, Zha, Carin. "Gromov-Wasserstein Learning for Graph Matching and Node Embedding." ICML 2019.** https://proceedings.mlr.press/v97/xu19b.html
- Seedless matching of graph structure through Gromov-Wasserstein (GW) distance. It works on fairly clean, highly overlapping graphs (for example, PPI networks and paired user–item graphs).
- FGWEA shows it does not work on real cross-source KGs, where neighbourhoods overlap only partly and are noisy.

**Seedless graph-alignment literature.**
- Spectral and optimal-transport methods such as GRASP (ACM TKDD 2023) and FUGAL (NeurIPS 2024; https://proceedings.neurips.cc/paper_files/paper/2024/hash/22b111819c74453837899689166c4cf9-Abstract.html) align a graph with a *noisy permuted copy of itself*.
- That setting is far easier than aligning two independently built ontologies.
- The EDBT 2023 "unrestricted graph alignment" evaluation could not be located (**UNVERIFIED**).

### D.3 LLM-based entity alignment

**Jiang, Shen, Shi, Xu, Li, Li, Guo, Shen, Wang. "Unlocking the Power of Large Language Models for Entity Alignment" (ChatEA). ACL 2024.** https://aclanthology.org/2024.acl-long.408
- Converts the KG into code-like text for the LLM, then aligns in two stages through dialogue.
- DBP15K EN-FR Hits@1 0.990, tying the name- and attribute-heavy BERT-INT. DBP–Wiki 0.995.
- Largest gains on the hard ICEWS–Wiki and ICEWS–YAGO pairs: +16% and +8.8% over the previous best.

**Yang, Chen, Wang, Yang, Wang, Liu. "Two Heads Are Better Than One: Integrating Knowledge from Knowledge Graphs and Large Language Models for Entity Alignment" (LLMEA). arXiv 2401.16960, 2024** (LREC-COLING 2024 venue **UNVERIFIED**). https://arxiv.org/abs/2401.16960
- Candidate generation from structural embeddings plus name edit distance, then the LLM answers multiple-choice questions to pick the match.
- Outperforms baselines on three datasets.

**Synthesis for D.**
- Structure alone, without seeds or names, essentially never aligns real knowledge graphs or ontologies: GW-only Hits@1 is about 0.01, matching our chance-level result.
- Even with 20% seeds, structure-only embedding EA reaches only about 0.25–0.5 Hits@1 on realistic splits.
- Names and text carry most of the signal. Plain string equality gets F1 0.77 on Anatomy, and names-only BERT gets 0.60–0.87 on DBP15K.
- Unsupervised name-based systems (PARIS, AML, LogMap) beat seeded embedding methods.
- The current best (2024–2026) combines lexical candidates, structure/logic for repair and context, and LLMs as judges.

---

## E. Scale: training systems, cost, and inductive or foundation models

**Lerer, Wu, Shen, Lacroix, Wehrstedt, Bose, Peysakhovich. "PyTorch-BigGraph: A Large-scale Graph Embedding System." SysML 2019.** https://arxiv.org/abs/1903.12287
- Full Freebase: 121M entities, 25k relations, 2.73B edges.
- TransE-like model, 10 epochs: **raw MRR 0.170, Hits@10 0.285**, scored against 10,000 *degree-sampled* negatives.
- Training time: 30 h on one machine, 7.7 h on 8 machines (MRR 0.163). Partitioning cuts memory by 88%.
- ComplEx reached MRR 0.24 on one partition but was unstable across partitions (0.15–0.22).
- Footnote: against *uniformly* sampled negatives, models reach more than 50% Hits@1 just by "ranking based on the degree distribution". At web scale, headline metrics can reflect node popularity rather than learned semantics.

**Zheng, Song, Ma, Tan, Ye, Dong, Xiong, Zhang, Karypis. "DGL-KE: Training Knowledge Graph Embeddings at Scale." SIGIR 2020** (arXiv:2004.08532; venue as commonly cited, not confirmed from the preprint: **UNVERIFIED**). https://arxiv.org/abs/2004.08532
- Freebase-86M: 86M nodes, 338M edges.
- Training time: 100 min on an 8-GPU EC2 instance, or 30 min on 4×48-core CPU machines. That is 2–5× faster than the best competitors.
- Reported MRR (TransE 0.743, ComplEx 0.716) uses only **2,000 unfiltered sampled negatives** (1,000 uniform plus 1,000 by degree), so it is not comparable to full-ranking MRR.

**Zhu, Xu, Qu, Tang. "GraphVite: A High-Performance CPU-GPU Hybrid System for Node Embedding." WWW 2019.** https://arxiv.org/abs/1903.00757
- About 1 minute for 1M nodes and 5M edges, about 20 h for 66M nodes and 1.8B edges, on 4 GPUs. About 50× faster than earlier systems.
- Also hosts the Wikidata5M benchmarks.

**Mohoney, Waleffe, Xu, Rekatsinas, Venkataraman. "Marius: Learning Massive Graph Embeddings on a Single Machine." OSDI 2021.** https://www.usenix.org/conference/osdi21/presentation/mohoney
- Freebase-86M with ComplEx (d=100) on a single GPU: MRR 0.726 in **2 h 1 min**, against 7 h 27 min for PBG (same 2,000-sampled-negatives evaluation).
- 10× faster than DGL-KE on Twitter.
- Trains more than 1B edges and 550 GB of parameters on one machine.

**Kochsiek, Gemulla. "Parallel Training of Knowledge Graph Embedding Models: A Comparison of Techniques." PVLDB 15(3):633–645, 2022.** https://doi.org/10.14778/3494124.3494144
- Prior evaluations of scalable KGE are "often not comparable and can be misleading".
- **Most parallel training methods hurt embedding quality.** Random partitioning with suitable sampling can be the best option.

**Zheng et al. "GraphStorm: All-in-one Graph Machine Learning Framework for Industry Applications." KDD 2024.** https://doi.org/10.1145/3637528.3671603 (arXiv:2406.06022)
- Scales to billion-node graphs and has been deployed in more than a dozen industry applications.
- Its focus is GNNs with node and edge features, not shallow KGE.

**OGB ogbl-wikikg2 leaderboard.** https://ogb.stanford.edu/docs/leader_linkprop/
- Plain KGE baselines: about 0.40–0.43 test MRR with about 1.25B parameters (TransE 0.426, RotatE 0.433, ComplEx 0.403). At 50–100 dimensions, about 0.25–0.38.
- The top entries (0.69–0.74) use NodePiece- or StarGraph-style compositional encoders, text (with external data), or ensembles.
- ogbl-biokg: ComplEx 0.810, RotatE 0.799, TransE 0.745. The best ensemble reaches 0.962.
- All results use **sampled-negative** MRR.

**Galkin, Denis, Wu, Hamilton. "NodePiece: Compositional and Parameter-Efficient Representations of Large Knowledge Graphs." ICLR 2022.** https://arxiv.org/abs/2106.12144
- Represents entities by anchors plus relation context, with under 10% of nodes as anchors and often 10× fewer parameters.
- On ogbl-wikikg2 it beats shallow KGE with **about 70× fewer parameters**.
- Inductive: can embed entities not seen in training.

**Galkin, Yuan, Mostafa, Tang, Zhu. "Towards Foundation Models for Knowledge Graph Reasoning" (ULTRA). ICLR 2024.** https://arxiv.org/abs/2310.04562
- A single pre-trained model that conditions on the graph of interactions between relations, so it works on *any* KG without retraining.
- Zero-shot average MRR across 57 graphs:
  - Inductive (27 graphs): 0.435, vs 0.342 for the supervised state of the art per graph.
  - Transductive (13 graphs): 0.312, vs 0.348.
  - Fine-tuned: 0.443 and 0.379.
- ULTRA transfers relational reasoning *patterns* across graphs whose vocabularies do not overlap.
- It does **not** align entities across graphs. It needs both graphs' structure, and it learns no shared entity identity.

**Synthesis for E.**
- Engineering is a solved problem: shallow KGE can be trained on hundreds of millions to billions of edges, in hours on one GPU (Marius) or a small cluster (PBG, DGL-KE).
- Quality at that scale is weak or unknown:
  - Full-ranking MRR on Wikidata-scale time splits is very low (0.18 on WikiKG90Mv2).
  - Headline large-scale numbers use sampled negatives and are partly explained by node degree.
  - Parallel training can reduce quality.
  - Text features gave the biggest jumps.
- For large or evolving ontologies, inductive models (NodePiece, ULTRA) avoid retraining and are the better-supported route. "Embeddings help performance on large graphs" is true for compute cost, not demonstrated for discovery quality.

---

## F. KGE vs LLMs and text-based KG completion (2019–2026)

**Yao, Mao, Luo. "KG-BERT: BERT for Knowledge Graph Completion." arXiv 2019.** https://arxiv.org/abs/1909.03193
- Cross-encoder triple scoring. Good recall, poor precision:
  - WN18RR: MRR 0.216, Hits@1 0.041, Hits@10 0.524.
  - FB15k-237: Hits@10 0.420.
- Very slow inference.

**Wang, Zhao, Wei, Liu. "SimKGC." ACL 2022.** https://arxiv.org/abs/2203.02167
- A bi-encoder trained with contrastive learning.

| Dataset | SimKGC MRR | Structure-based comparison | Notes |
|---|---|---|---|
| WN18RR | **0.666–0.671** (H@1 0.587) | RotatE 0.476, TuckER 0.470 | Text wins by about 0.2 MRR |
| FB15k-237 | 0.336 | TuckER 0.358 | Text does *not* win |
| Wikidata5M-Trans | 0.358 | RotatE 0.290 | |
| Wikidata5M-Ind | 0.714 | — | Embeddings cannot do this at all |

- **Interpretation:** WN18RR is lexical. Many of its relations (derivationally_related_form, hypernym) link synsets whose definitions share words. FB15k-237 is dense (about 37 edges per entity) and rewards learned inference rules, where structure models and path methods hold up.
- Also gives the human-evaluation evidence on false negatives (section C).

**Wang et al. StAR (WWW 2021)** is cited only through the SimKGC and DIFT tables: WN18RR MRR 0.401–0.551, FB15k-237 0.296–0.365, depending on the re-run.

**Wei, Huang, Zhang, Kwok. "KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion." Findings of EMNLP 2023.** https://aclanthology.org/2023.findings-emnlp.580
- A KGE retriever proposes candidates and ChatGPT re-ranks them using demonstrations from the graph.
- FB15k-237 MRR **0.412** (H@1 0.327). WN18RR 0.549 (H@1 0.474).
- Plain **ChatGPT zero/one-shot**: FB15k-237 H@1 0.237 / 0.267, and WN18RR H@1 0.190 / 0.212. That is *below* RotatE's H@1 of 0.241 / 0.428.
- Without in-context learning, KICGPT drops to 0.342 MRR.

**Liu, Tian, Sun, Hu. "Finetuning Generative Large Language Models with Discrimination Instructions for Knowledge Graph Completion" (DIFT). ISWC 2024.** https://arxiv.org/abs/2407.16127
- A fine-tuned LLaMA picks the answer from a KGE or text model's candidate list.

| Method | FB15k-237 MRR | WN18RR MRR |
|---|---|---|
| DIFT + CoLE | **0.439** | 0.617 |
| DIFT + SimKGC | 0.402 | **0.686** |
| NBFNet (structure-only GNN) | 0.415 | 0.551 |
| LLaMA re-ranker without fine-tuning | about 0.23 | — |

- The LLM helps as a **re-ranker on top of a structural or text retriever**. On its own it is weak.

**Zhu, Wang, Chen et al. "LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities" (AutoKG). arXiv 2305.13168** (published in World Wide Web journal 2024: venue **UNVERIFIED**). https://arxiv.org/abs/2305.13168
- On small samples, GPT-4 "matches or nears" the state of the art for FB15k-237 link prediction.
- When KICGPT's authors re-ran this on the whole test set, ChatGPT reached H@1 0.24–0.27, below supervised models.
- LLMs are weaker at KG *construction* than fine-tuned extractors.

**Jiang, Cao, Xiao, Bhatia, Sun, Han. "KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge." NeurIPS 2024.** https://arxiv.org/abs/2405.16412
- An LLM builds an entity-cluster hierarchy from text. KGE models (TransE, RotatE and others) are then fine-tuned to respect that hierarchy.
- Hits@10 gains over the base structure-only KGE: +12.6% (FB15k-237), +6.7% (YAGO3-10), +17.7% (PrimeKG). Gains over the best PLM-based methods: +14.4%, +13.5%, +11.9%.
- **Complementarity** is the main lesson of the 2023–2025 work: LLM semantics plus KG structure beats either alone.

**Yao, Peng, Mao, Luo. "Exploring Large Language Models for Knowledge Graph Completion" (KG-LLM). ICASSP 2025.** https://github.com/yao8839836/kg-llm (arXiv:2308.13916)
- Instruction-tuned LLMs for triple classification and relation prediction.
- Strong on those *classification* tasks, but they are easier than full entity ranking. Numbers not extracted.

**Synthesis for F.**
- On lexically rich graphs like WN18RR, which most resembles an ontology, text-based models clearly beat structural KGE: MRR 0.67 vs 0.48.
- On dense, rule-rich graphs like FB15k-237, structure and path methods are on par or better.
- LLMs used directly as link predictors underperform tuned KGE.
- The best 2023–2025 systems are hybrids: a KGE or text retriever plus an LLM re-ranker (KICGPT, DIFT), or a KGE trained under LLM-derived structure (KG-FIT).
- LLM benchmark scores may be inflated by memorisation of Freebase/Wikidata facts from pre-training. This is a widely noted concern; no definitive quantitative study was found (**UNVERIFIED**).

---

## Implications for the colleague's four claims

| Claim | What the literature says | Verdict |
|---|---|---|
| **1. KGE discovers real missing relationships** | Partly true, with low precision. Time-split studies recover about 24% of future therapeutic links in the top 200 (Rosalind). About 49% of "wrong" top-1 predictions on Wikidata5M were judged correct, mostly on generic 1-to-N relations. But plausible-but-false predictions are common (CoDEx). Results depend on the random seed (only 2–4 of the top-10 shared across seeds). On real ontologies, standard KGE subsumption prediction is very poor (MRR 0.015–0.08 on GO and FoodOn), and even purpose-built EL++ embeddings almost never rank *entailed* subsumptions first (H@1 ≈ 0.00–0.01). Many "discoveries" in an ontology are things a reasoner already entails. | **Weak / conditional.** Use KGE output as a candidate list to be filtered by a reasoner and an expert, and evaluate it on a later release. |
| **2. Useful suggested connections** | Usefulness has been shown only with heavy curation (Rosalind: 9% wet-lab hit rate after expert triage). Unstable across seeds; decent MRR does not imply stable top-k. Strong rule and path baselines (AnyBURL, SAFRAN, NBFNet) match or beat KGE and give explanations. Frequency baselines explain much of FB15k-237. | **Unproven for ontologies.** Prefer rule or path methods (explainable), ensembles across seeds, and reasoner-filtered evaluation. |
| **3. Helps performance on large graphs** | True for compute: billions of edges in hours (PBG, DGL-KE, Marius). Not shown for quality: sampled-negative metrics inflate scores, node degree explains much of the "accuracy", parallel training can hurt quality, and full-ranking MRR on WikiKG90Mv2 is 0.18. Inductive or foundation models (NodePiece, ULTRA) and text features gave the real gains. | **True for throughput only.** Quality at scale needs text or inductive models and honest full-ranking evaluation. |
| **4. "Naturally crosses semantic taxonomies"** | Contradicted. Structure-only alignment without seeds or names is at chance (FGWEA GW-only Hits@1 0.004–0.011; our zero-anchor result). Even with 20% seeds, structure-only EA gets Hits@1 of about 0.25–0.5. Names and text dominate: string equality gives F1 0.77 on OAEI Anatomy, names-only BERT gives 0.60–0.87 on DBP15K, and "unsupervised" SelfKG is really LaBSE name embeddings. Best practice (2023–2026) is lexical candidates, structure or logic for repair, and LLMs as judges (LogMap+LLM, MILA, OLaLa). Independently trained KGE spaces are not comparable without anchors, and ULTRA transfers reasoning patterns, not entity identity. | **False as stated.** Cross-taxonomy mapping needs lexical/LLM signals and/or seed anchors; KGE structure is at most a secondary refinement signal. |

### Recommended framing for the report

- KGE is a reasonable, scalable *candidate generator*. On ontology graphs it should be:
  - benchmarked against strong rule and path baselines (AnyBURL or SAFRAN, NBFNet) and a frequency baseline;
  - evaluated with entailed edges filtered out (deductive closure);
  - validated on a later ontology release (for example GO or SNOMED N → N+1);
  - run over several seeds;
  - combined with text (label and definition embeddings) and a DL reasoner.
- For mapping between two vocabularies, the evidence-based pipeline is:
  1. Lexical and embedding candidate generation (labels, synonyms, LLM sentence embeddings).
  2. Structural or logical extension and repair (LogMap, AML).
  3. Optionally, an LLM as judge for uncertain pairs.
  Expect plain string matching alone to recover most easy mappings.

---

## Reference index (by topic)

**A.**
- Sun et al. ACL 2020
- Ruffinelli et al. ICLR 2020
- Rossi et al. TKDD 2021
- Ali et al. TPAMI 2021
- Kadlec et al. 2017
- Toutanova & Chen 2015
- Akrami et al. SIGMOD 2020
- Meilicke et al. IJCAI 2019 / VLDB J 2024
- Ott et al. AKBC 2021
- Zhu et al. NeurIPS 2021
- Safavi & Koutra EMNLP 2020
- Yang et al. NeurIPS 2022
- Cao et al. ACL 2021
- Hoyt et al. GLB 2022
- Méroué et al. ESWC 2026
- Bonner et al. AILSCI 2022
- Sun et al. ICLR 2019
- Ali et al. JMLR 2021

**B.**
- Chen et al. MLJ 2021
- Kulmanov et al. IJCAI 2019
- Jackermeier et al. WWW 2024
- Mashkova et al. NeSy 2024
- Chen et al. TKDE 2025
- Chen et al. WWWJ 2023
- He et al. Findings ACL 2023
- He et al. SWJ 2024
- He et al. NeurIPS 2024 (HiT, "Language Models as Hierarchy Encoders": https://arxiv.org/abs/2401.11374; re-trains LM encoders into hyperbolic hierarchy encoders and beats pre-trained/fine-tuned LMs and hyperbolic baselines on transitive inference and subsumption)
- Li et al. 2024
- Sun et al. PVLDB 2024
- Nickel & Kiela NeurIPS 2017
- Ganea et al. ICML 2018
- Balažević et al. NeurIPS 2019
- Chami et al. ACL 2020
- Gutiérrez-Basulto & Schockaert KR 2018
- Song et al. NeurIPS 2021

**C.**
- Paliwal et al. Sci Rep 2020
- Himmelstein et al. eLife 2017
- Zeng et al. JPR 2020
- Huang et al. Nat Med 2024
- Pu et al. JBI 2023
- Hu et al. NeurIPS 2020 (OGB)
- Hu et al. NeurIPS D&B 2021 (OGB-LSC)
- Wang et al. ACL 2022 (SimKGC)
- Dong et al. CIKM 2023
- Zhou et al. Genome Biology 2019

**D.**
- OAEI Anatomy results 2021–2024
- He et al. AAAI 2022
- He et al. ISWC 2022
- He et al. ISWC 2023 P&D
- Hertling & Paulheim K-CAP 2023
- Qiang et al. PVLDB 2025
- Taboada et al. Inf. Fusion 2025
- Lushnei et al. EACL 2026
- Sun et al. PVLDB 2020 (OpenEA)
- Chen et al. IJCAI 2017
- Wang et al. EMNLP 2018
- Sun et al. IJCAI 2018
- Wu et al. IJCAI 2019
- Liu et al. EMNLP 2020
- Liu et al. WWW 2022
- Tang et al. Findings ACL 2023
- Xu et al. ICML 2019
- FUGAL NeurIPS 2024
- Jiang et al. ACL 2024
- Yang et al. 2024 (LLMEA)

**E.**
- Lerer et al. SysML 2019
- Zheng et al. 2020 (DGL-KE)
- Zhu et al. WWW 2019
- Mohoney et al. OSDI 2021
- Kochsiek & Gemulla PVLDB 2022
- Zheng et al. KDD 2024
- OGB leaderboard
- Galkin et al. ICLR 2022
- Galkin et al. ICLR 2024

**F.**
- Yao et al. 2019 (KG-BERT)
- Wang et al. ACL 2022
- Wei et al. Findings EMNLP 2023
- Liu et al. ISWC 2024
- Zhu et al. 2023/2024 (AutoKG)
- Jiang et al. NeurIPS 2024
- Yao et al. ICASSP 2025

Total: about 75 distinct sources, all located during this survey. Items with an uncertain venue or unextracted numbers are marked **UNVERIFIED** in the text.
