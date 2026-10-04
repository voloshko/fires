#!/usr/bin/env bash
# SPEC-114: six matched, adapter-only jobs; serialize each GPU lease.
set -euo pipefail
cd /home/mc/fires-accuracy-next
for seed in 1 2 3; do
  for variant in pair post; do
    (
      exec 9>research/accuracy-gpu.lock
      flock 9
      while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if test "$free" -ge 22000; then break; fi
        sleep 30
      done
      .runtime/bin/python scripts/train_equal_cost_additions.py --data-root data-store \
        --reference-root /home/mc/fires/research --runs research/equal-cost-additions-v1 \
        --seed "$seed" --variant "$variant"
    )
  done
done
