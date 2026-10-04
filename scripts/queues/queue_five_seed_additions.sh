#!/usr/bin/env bash
# SPEC-118: six matched, adapter-only jobs; serialize each GPU lease.
set -euo pipefail
cd /home/mc/fires-accuracy-next
for seed in 4 5; do
  for variant in pair post; do
    (
      exec 9>research/accuracy-gpu.lock
      flock 9
      while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if test "$free" -ge 22000; then break; fi
        sleep 30
      done
      .runtime/bin/python scripts/train_five_seed_additions.py --data-root data-store \
        --reference-root /home/mc/fires/research --runs research/five-seed-additions-v1 \
        --seed "$seed" --variant "$variant"
    )
  done
done
