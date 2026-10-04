#!/usr/bin/env bash
# SPEC-114: one real mechanical GPU test under the existing shared lease.
set -euo pipefail
cd /home/mc/fires-accuracy-next
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
.runtime/bin/python scripts/smoke_equal_cost_additions.py --data-root data-store \
  --base /home/mc/fires/research/hls-c1mm-final-s1/model.pt \
  --out research/equal-cost-additions-smoke-v1.json
