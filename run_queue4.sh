#!/usr/bin/env bash
# Fourth batch: ensembles, then the rest of the model zoo (resumes; finished models are skipped).
set -u
cd "$(dirname "$0")"
until grep -q QUEUE3_DONE results/exp8.log 2>/dev/null; do sleep 30; done
uv run python experiments/exp13_ensembles.py > results/exp13.log 2>&1
uv run python experiments/exp7_model_zoo.py > results/exp7_resume.log 2>&1
echo QUEUE4_DONE >> results/exp13.log
