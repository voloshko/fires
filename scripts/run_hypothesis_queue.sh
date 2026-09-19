#!/usr/bin/env bash
# SPEC-36: wait for the existing campaign; never preempt another experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
waiting=1
while (( waiting )); do
  waiting=0
  for unit in run-queue13 run-queue14 run-smp3 run-smp4 run-f0 run-f1 run-f2 run-f3 run-f4 run-rn1 run-rn2 run-long1 run-long2 run-inf13; do
    if systemctl --user is-active --quiet "$unit"; then waiting=1; fi
  done
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
  if (( free_mb < 22000 )); then waiting=1; fi
  if (( waiting )); then sleep 30; fi
done
printf 'GPU ready at %s\n' "$(date --iso-8601=seconds)"
# All commands run one at a time and keep their own artifacts/logs.
run() {
  local tag="$1"; shift
  if [[ -f "research/$tag/summary.json" ]]; then return; fi
  .venv/bin/python scripts/wait_hypothesis_gpu.py
  "$@" > "logs/$tag.log" 2>&1
}
run af-net-v1 .venv/bin/python scripts/hypothesis_af_net.py --out research/af-net-v1
while systemctl --user is-active --quiet fires-hyp-bs-boost; do sleep 20; done
[[ -f research/bs-boost-v1/probabilities.npy ]]
for seed in 20260918 20260919; do
  for variant in optical raw siam; do
    suffix=v1
    if [[ "$variant" == siam ]]; then suffix=v2; fi
    tag="bs-$variant-$seed-$suffix"
    run "$tag" .venv/bin/python scripts/hypothesis_bs.py train --variant "$variant" --seed "$seed" --out "research/$tag"
  done
done
printf 'Queue complete at %s\n' "$(date --iso-8601=seconds)"
