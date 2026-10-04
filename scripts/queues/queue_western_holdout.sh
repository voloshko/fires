#!/usr/bin/env bash
# SPEC-105: only admitted, completely collected independent geography may score.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-western-collect-v1; do sleep 15; done
.runtime/bin/python - <<'PY'
import json
x=json.load(open("data-store/western-holdout-v1/manifest.json"))
assert x["spec_id"]=="SPEC-105" and x["ready_for_evaluation"] and not x["qc_probe"]
assert x["counts"]["samples"]>=50 and x["counts"]["groups"]>=20
assert x["selection_file"]=="selection-v2.json"
PY
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
.runtime/bin/python scripts/score_western_holdout.py --data-root data-store \
  --manifest data-store/western-holdout-v1/manifest.json --runs research/russian-transfer-v1 \
  --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars \
  --parent research/russian-transfer-v1-result.json --out research/western-holdout-v1-result.json
