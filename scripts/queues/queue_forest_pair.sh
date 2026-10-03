#!/usr/bin/env bash
# SPEC-91: one owned GPU job; freeze completed inputs before any training.
set -euo pipefail
cd /home/mc/fires-accuracy-next
PY=.runtime/bin/python
MANIFEST=data-store/forest-pairs/manifest.json
exec 9>research/accuracy-gpu.lock
flock -n 9 || { echo 'Another accuracy campaign owns the GPU queue'; exit 2; }
while systemctl --user is-active --quiet fires-accuracy-pairs-k8-v1; do sleep 30; done
test -f "$MANIFEST"
"$PY" -c 'import json; x=json.load(open("data-store/forest-pairs/manifest.json")); assert x["ready_for_training"] and not x["qc_probe"]'
for seed in 1 2 3; do
  for variant in post pair; do
    out="research/forest-pair-v1/${variant}-s${seed}"
    if test -f "$out/summary.json"; then
      "$PY" -c 'import json,sys; from src.comp.accuracy_contract import sha256_file; x=json.load(open(sys.argv[1]+"/summary.json")); assert x["completed"] and not x["smoke"]; assert x["seed"]==int(sys.argv[2]) and x["variant"]==sys.argv[3]; assert x["data_manifest"]==sha256_file("data-store/forest-pairs/manifest.json")' "$out" "$seed" "$variant"
      continue
    fi
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
      if test "$free" -ge 22000; then break; fi
      sleep 30
    done
    "$PY" scripts/forest_pair_train.py --manifest "$MANIFEST" --data-root data-store \
      --out "$out" --variant "$variant" --seed "$seed"
  done
done
echo 'Six matched training runs completed; acceptance scoring is still required.'
