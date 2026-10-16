#!/usr/bin/env bash
# Runs the remaining experiments sequentially (they share the GPU; serial runs keep timings clean).
set -u
cd "$(dirname "$0")"
while pgrep -f exp1_link_prediction.py >/dev/null; do sleep 10; done
uv run python experiments/exp3_alignment.py --dataset wn18rr --epochs 40 > results/exp3_wn18rr.log 2>&1
uv run python experiments/exp3_alignment.py --dataset fb15k237 --epochs 20 --fractions 0.05,0.1,0.3 > results/exp3_fb15k237.log 2>&1
uv run python experiments/exp2_scaling.py > results/exp2.log 2>&1
echo ALL_DONE >> results/exp2.log
