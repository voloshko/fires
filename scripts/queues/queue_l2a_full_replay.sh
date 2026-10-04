#!/usr/bin/env bash
# SPEC-101: matched full-pool jobs; release GPU between jobs for other scoring.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-full-replay-prepare-v2 || \
      systemctl --user is-active --quiet fires-accuracy-full-replay-smoke-v2; do sleep 20; done
.runtime/bin/python - <<'PY'
import json
from pathlib import Path
summaries={}
for variant in ("control","replay"):
    path=Path(f"research/l2a-full-replay-smoke-{variant}-s1/summary.json")
    x=json.load(path.open())
    assert x["completed"] and x["smoke"] and x["epochs"]==2
    assert len(x["full_hls_exposures"])==3 and min(x["full_hls_exposures"])>=1
    summaries[variant]=x
assert summaries["control"]["order_sha256"]==summaries["replay"]["order_sha256"]
assert summaries["control"]["common_exposures"]==summaries["replay"]["common_exposures"]
assert not any(summaries["control"]["l2a_exposures"])
assert any(summaries["replay"]["l2a_exposures"])
PY
seal() {
.runtime/bin/python - <<'PY'
import json
from src.comp.accuracy_artifacts import execution_sources,freeze_recipe
from src.comp.accuracy_contract import sha256_file
manifest="data-store/l2a-full-replay-v1/manifest-v2.json"
x=json.load(open(manifest));assert x["ready_for_replay"] and x["full_original_fit_count"]==929
sources=execution_sources("scripts/l2a_full_replay_train.py","scripts/l2a_full_replay_io.py",
    "src/comp/l2a_full_replay.py","src/comp/l2a_finetuning.py","scripts/queues/queue_l2a_full_replay.sh",tta=False)
config=dict(manifest=sha256_file(manifest),epochs=30,batch=8,seeds=[1,2,3],variants=["control","replay"],
    initializers=[r["sha256"] for r in x["initializers"]],lr=3e-5,schedule="odd-full/even-common/fallback-full")
print("RECIPE",freeze_recipe("research/l2a-full-replay-v1/recipe.json",sources,config),flush=True)
PY
}
gpu_ready() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if test "$free" -ge 22000; then break; fi
    sleep 30
  done
}
exec 9>research/accuracy-gpu.lock
seal
for seed in 1 2 3; do
  for variant in control replay; do
    out="research/l2a-full-replay-v1/${variant}-s${seed}"
    if test -e "$out"; then echo "Immutable run exists: $out"; exit 2; fi
    flock 9
    gpu_ready
    seal
    .runtime/bin/python scripts/l2a_full_replay_train.py --data-root data-store \
      --manifest data-store/l2a-full-replay-v1/manifest-v2.json \
      --checkpoint "/home/mc/fires/research/hls-c1mm-final-s${seed}/model.pt" \
      --seed "$seed" --variant "$variant" --out "$out"
    flock -u 9
  done
done
flock 9
gpu_ready
seal
.runtime/bin/python scripts/score_l2a_full_replay.py --data-root data-store \
  --manifest data-store/l2a-full-replay-v1/manifest-v2.json --runs research/l2a-full-replay-v1 \
  --reference-root /home/mc/fires/research --out research/l2a-full-replay-v1-result.json
