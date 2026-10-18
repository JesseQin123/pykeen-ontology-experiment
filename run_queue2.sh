#!/usr/bin/env bash
# Second batch of GPU experiments, run serially after the GO temporal run.
set -u
cd "$(dirname "$0")"
while pgrep -f exp4_go_temporal.py >/dev/null; do sleep 15; done
uv run python experiments/exp5_oaei_anatomy.py > results/exp5.log 2>&1
uv run python experiments/exp7_model_zoo.py > results/exp7.log 2>&1
uv run python experiments/exp4_go_temporal.py --old 2018-01-01 --new 2020-01-01 --llm_n 0 > results/exp4_2018_2020.log 2>&1
echo QUEUE2_DONE >> results/exp7.log
