#!/usr/bin/env bash
# SPEC-103: standalone confirmation takes the GPU between full-replay jobs.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-component-development-v1; do sleep 10; done
.runtime/bin/python - <<'PY'
import json
x=json.load(open("research/component-rescue-v1-plan.json"))
assert x["eligible"] and x["spec_id"]=="SPEC-103"
PY
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
.runtime/bin/python scripts/confirm_component_rescue.py --data-root data-store \
  --plan research/component-rescue-v1-plan.json --development research/component-rescue-v1-development.json \
  --parent research/russian-transfer-v1-result.json --russian-manifest data-store/russia/manifest.json \
  --runs research/russian-transfer-v1 --reference-root /home/mc/fires/research \
  --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars \
  --out research/component-rescue-v1-confirmation.json
