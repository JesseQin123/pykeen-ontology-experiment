#!/usr/bin/env bash
# Frontier-LLM chain: after exp15's sampled runs: full-set reranking, blind rating (3 judges), LLM-on-structure.
set -u
cd "$(dirname "$0")"
until grep -q EXP15_DONE results/exp15_c.log 2>/dev/null; do sleep 30; done
uv run python experiments/exp15_frontier_llm.py rerank --sets go_novel_isa_full,go_new_terms_full > results/exp15_full.log 2>&1
uv run python experiments/exp16_llm_blind_rating.py > results/exp16.log 2>&1
uv run python experiments/exp17_llm_on_structure.py B A > results/exp17.log 2>&1
echo LLM_CHAIN_DONE >> results/exp17.log
