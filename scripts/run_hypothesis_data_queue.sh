#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet fires-hyp-external-train; do sleep 30; done
if [[ -f research/external-train-v1/download.json && ! -f research/external-train-v1/prepared-v2.json ]]; then
  .venv/bin/python scripts/hypothesis_external.py prepare > logs/external-prepare.log 2>&1
fi
while systemctl --user is-active --quiet fires-hyp-gpu-queue; do sleep 30; done
# A failed upstream run is not evidence that the GPU is available to this stage.
[[ -f research/bs-siam-20260919-v1/summary.json ]]
run() { local tag="$1"; shift; .venv/bin/python scripts/wait_hypothesis_gpu.py; "$@" > "logs/$tag.log" 2>&1; }
if .venv/bin/python -c 'import json; assert json.load(open("research/temporal-pre-v1/result.json"))["accepted"] >= 4'; then
  for seed in 20260918 20260919; do
    tag="bs-temporal-$seed-v1"
    run "$tag" .venv/bin/python scripts/hypothesis_bs.py train --variant optical --seed "$seed" --extra research/temporal-pre-v1 --out "research/$tag"
  done
fi
if .venv/bin/python -c 'import json; assert json.load(open("research/external-train-v1/prepared-v2.json"))["eligible"]'; then
  run external-pretrain .venv/bin/python scripts/hypothesis_external.py pretrain
  for seed in 20260918 20260919; do
    tag="bs-external-$seed-v1"
    run "$tag" .venv/bin/python scripts/hypothesis_bs.py train --variant siam --seed "$seed" --encoder research/external-train-v1/pretrain/encoder.pt --out "research/$tag"
  done
fi
printf 'Data hypothesis queue complete at %s\n' "$(date --iso-8601=seconds)"
