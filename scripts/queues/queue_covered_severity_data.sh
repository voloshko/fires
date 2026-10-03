#!/usr/bin/env bash
# SPEC-100: require a real clean QC pair before the full fixed collection.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-covered-qc-v1; do sleep 10; done
status=$(systemctl --user show fires-accuracy-covered-qc-v1 -p ExecMainStatus --value)
test "$status" = 0
.runtime/bin/python - <<'PY'
import json
from src.comp.accuracy_contract import sha256_file
x=json.load(open("data-store/severity-covered-v1/qc-manifest.json"))
assert x["spec_id"]=="SPEC-100" and x["qc_probe"] and x["counts"]["samples"]>0
assert x["selection_sha256"]==sha256_file("data-store/severity-covered-v1/selection.json")
PY
.runtime/bin/python scripts/prepare_fresh_paired_severity.py --profile pc-covered-2020-24 \
  --data-root data-store --source-root /home/mc/fires --prior data-store/prior-footprints.json \
  --out data-store/severity-covered-v1
