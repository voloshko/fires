#!/usr/bin/env bash
# SPEC-119: fresh confirmation uses the same shared GPU lease/admission.
set -euo pipefail
cd /home/mc/fires-accuracy-next
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
.runtime/bin/python scripts/score_five_seed_eastern.py --data-root data-store \
  --selection data-store/eastern-holdout-plan-v1/selection.json \
  --candidate research/five-seed-additions-v1/development.json \
  --runs research/five-seed-additions-v1 --reference-root /home/mc/fires/research \
  --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars \
  --manifest data-store/eastern-confirmation-v3/manifest.json \
  --out research/eastern-confirmation-v3/result.json
