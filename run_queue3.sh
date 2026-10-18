#!/usr/bin/env bash
# Third batch: waits for queue 2, then scale-accuracy (GPU). Poincare runs on CPU separately.
set -u
cd "$(dirname "$0")"
until grep -q QUEUE2_DONE results/exp7.log 2>/dev/null; do sleep 30; done
uv run python experiments/exp4_go_temporal.py > results/exp4_2020_2026_v2.log 2>&1
uv run python experiments/exp10_new_terms.py > results/exp10.log 2>&1
uv run python experiments/exp8_scale_accuracy.py > results/exp8.log 2>&1
uv run python experiments/exp5b_oaei_ablation.py > results/exp5b.log 2>&1
echo QUEUE3_DONE >> results/exp8.log
