#!/usr/bin/env bash
# SPEC-116: fresh confirmation uses the same shared GPU lease/admission.
set -euo pipefail
cd /home/mc/fires-accuracy-next
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
.runtime/bin/python scripts/score_equal_cost_eastern.py --data-root data-store \
  --selection data-store/eastern-holdout-plan-v1/selection.json \
  --candidate research/equal-cost-additions-v1/development.json \
  --runs research/equal-cost-additions-v1 --reference-root /home/mc/fires/research \
  --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars \
  --manifest data-store/eastern-confirmation-v2/manifest.json \
  --out research/eastern-confirmation-v2/result.json
