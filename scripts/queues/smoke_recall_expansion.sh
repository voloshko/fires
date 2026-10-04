#!/usr/bin/env bash
# SPEC-111: one real mechanical GPU test under the existing shared lease.
set -euo pipefail
cd /home/mc/fires-accuracy-next
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
.runtime/bin/python scripts/smoke_recall_expansion.py --data-root data-store \
  --base /home/mc/fires/research/hls-c1mm-final-s1/model.pt \
  --out research/recall-expansion-smoke-v1.json
