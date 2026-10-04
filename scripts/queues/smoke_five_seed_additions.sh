#!/usr/bin/env bash
# SPEC-118: one real mechanical GPU test under the existing shared lease.
set -euo pipefail
cd /home/mc/fires-accuracy-next
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
for seed in 4 5; do
  .runtime/bin/python scripts/smoke_five_seed_additions.py --data-root data-store \
    --reference-root /home/mc/fires/research --seed "$seed" \
    --out "research/five-seed-additions-smoke-s${seed}-v1.json"
done
