#!/usr/bin/env bash
# SPEC-91: score only after all paired training artifacts exist.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-train-k8-v1; do sleep 30; done
exec 9>research/accuracy-gpu.lock
flock 9
for variant in post pair; do
  for seed in 1 2 3; do test -f "research/forest-pair-v1/${variant}-s${seed}/summary.json"; done
done
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
.runtime/bin/python scripts/score_forest_pair.py --data-root data-store \
  --manifest data-store/forest-pairs/manifest.json --runs research/forest-pair-v1 \
  --reference-root /home/mc/fires/research --out research/forest-pair-v1-result.json
