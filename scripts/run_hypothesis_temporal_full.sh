#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f research/temporal-full-catalog-v1/catalog.json ]] || .venv/bin/python scripts/hypothesis_sources.py temporal --limit 144 --out research/temporal-full-catalog-v1
[[ -f research/temporal-full-pre-v2/result.json ]] || .venv/bin/python scripts/hypothesis_temporal.py --catalog research/temporal-full-catalog-v1 --out research/temporal-full-pre-v2
.venv/bin/python -c 'import json; d=json.load(open("research/temporal-full-pre-v2/result.json")); print("FULL TEMPORAL QC",d["accepted"]); assert d["accepted"] >= 50'
while systemctl --user is-active --quiet fires-hyp-gpu-queue || systemctl --user is-active --quiet fires-hyp-data-queue-v2; do sleep 30; done
[[ -f research/bs-siam-20260919-v2/summary.json ]]
for seed in 20260918 20260919; do
  .venv/bin/python scripts/wait_hypothesis_gpu.py
  .venv/bin/python scripts/hypothesis_bs.py train --variant optical --seed "$seed" --extra research/temporal-full-pre-v2 --min-extra 50 --out "research/bs-temporal-full-$seed-v1" > "logs/bs-temporal-full-$seed-v1.log" 2>&1
done
