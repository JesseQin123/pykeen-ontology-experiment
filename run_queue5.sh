#!/usr/bin/env bash
# Fifth batch: corrected scale-accuracy run, after queue 4.
set -u
cd "$(dirname "$0")"
until grep -q QUEUE4_DONE results/exp13.log 2>/dev/null; do sleep 30; done
uv run python experiments/exp8_scale_accuracy.py > results/exp8.log 2>&1
echo QUEUE5_DONE >> results/exp8.log
