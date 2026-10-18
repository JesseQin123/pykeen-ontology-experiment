# Graph embeddings on semantic ontology graphs: a validation

Does PyKEEN-style knowledge graph embedding actually help on ontology graphs? 17 experiments test four claims,
each against non-embedding baselines (popularity, AnyBURL rules, an is_a reasoner, label-text embeddings,
anchor-neighbourhood matching) and with frontier LLMs as re-rankers, generators and blind judges:

1. Embeddings recover relations the graph is missing: random hold-out (WN18RR) and a prospective test that trains on
   Gene Ontology 2020 and scores what curators added by 2026.
2. Embedding-suggested links are useful: seed stability, three-judge LLM blind rating, and a candidates -> LLM-filter pipeline.
3. Embeddings help on large graphs: query latency vs. size, ANN recall, and accuracy vs. graph size on YAGO3-10.
4. Embeddings align two ontologies with different terms: renamed views, seed-free optimal transport, and the real
   OAEI Anatomy track (mouse vs. human) scored against the official reference.

Results and the converged recommendations (Chinese, with an English summary): [REPORT.md](REPORT.md).
All quoted numbers trace to `results/SUMMARY.json`, `results/STATS.json` and the per-experiment JSON files.

```bash
uv sync
uv run python experiments/exp4_go_temporal.py      # see REPORT.md for the full list
```

Training runs locally (Apple MPS). LLM and embedding inference run on hosted endpoints configured in a
gitignored `.env` (`RELAY_BASE_URL`, `RELAY_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_API_KEY`).
