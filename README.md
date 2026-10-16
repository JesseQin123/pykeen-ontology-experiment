# Graph embeddings on semantic ontology graphs: a validation

Does PyKEEN-style knowledge graph embedding actually help on ontology graphs? This repo tests four claims
against non-embedding baselines (popularity, an AnyBURL-style path-rule learner, anchor-neighbourhood matching):

1. Embeddings recover relations that explicit graph search misses (WN18RR link prediction, reachability split).
2. Embedding-suggested links are real (open-world check against full WordNet 3.0).
3. Embeddings speed up queries on large graphs (k-hop traversal vs. brute-force / HNSW kNN, training cost).
4. Embeddings align two ontologies that use different terms (renamed graph views, seed anchors, synonym labels).

Results and discussion (Chinese, with an English summary): [REPORT.md](REPORT.md).

```bash
uv sync
uv run python experiments/exp1_link_prediction.py
uv run python experiments/exp3_alignment.py --dataset wn18rr
uv run python experiments/exp3_alignment.py --dataset fb15k237 --epochs 20 --fractions 0.05,0.1,0.3
uv run python experiments/exp2_scaling.py
uv run python experiments/exp2b_ann_realistic.py
```
