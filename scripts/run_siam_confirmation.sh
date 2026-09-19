#!/usr/bin/env bash
# SPEC-36: new seeds, complete-event folds, identical precision and boost.
set -euo pipefail
cd "$(dirname "$0")/.."
for fold in 0 1 2 3 4; do
  dir="research/bs-confirm-boost-f$fold-v1"
  if [[ ! -f "$dir/probabilities.npy" ]]; then
    .venv/bin/python scripts/hypothesis_bs.py boost --fold "$fold" --out "$dir" > "logs/bs-confirm-boost-f$fold.log" 2>&1
  fi
done
while systemctl --user is-active --quiet fires-hyp-gpu-queue || systemctl --user is-active --quiet fires-hyp-data-queue-v2 || systemctl --user is-active --quiet fires-hyp-temporal-full-v2; do sleep 30; done
[[ -f research/bs-siam-20260919-v3/summary.json ]]
for fold in 0 1 2 3 4; do
  seed=$((20260920+fold))
  for variant in optical siam; do
    dir="research/bs-confirm-$variant-f$fold-v1"
    if [[ -f "$dir/summary.json" ]]; then continue; fi
    .venv/bin/python scripts/wait_hypothesis_gpu.py
    .venv/bin/python scripts/hypothesis_bs.py train --fold "$fold" --seed "$seed" --variant "$variant" --precision bf16 --boost "research/bs-confirm-boost-f$fold-v1" --out "$dir" > "logs/bs-confirm-$variant-f$fold.log" 2>&1
  done
done
