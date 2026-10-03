#!/usr/bin/env bash
# SPEC-94: matched class balancing, with a code/recipe seal before any start.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-train-k8-v1 || \
      systemctl --user is-active --quiet fires-accuracy-pair-score-v1 || \
      systemctl --user is-active --quiet fires-accuracy-severity-prepare-v1; do sleep 30; done
test -f data-store/severity-pairs/manifest.json
.runtime/bin/python -c 'import json; x=json.load(open("data-store/severity-pairs/manifest.json")); assert x["ready_for_training"] and not x["qc_probe"]'
exec 9>research/accuracy-gpu.lock
flock 9
seal() {
  .runtime/bin/python - <<'PY'
from pathlib import Path
from src.comp.accuracy_artifacts import execution_sources, freeze_recipe
from src.comp.accuracy_contract import sha256_file
sources = execution_sources("scripts/paired_severity_train.py", "src/comp/paired_severity.py",
                            "scripts/forest_pair_train.py", "scripts/queues/queue_paired_severity.sh", tta=False)
config = dict(manifest=sha256_file("data-store/severity-pairs/manifest.json"), epochs=120,
              width=32, depth=7, batch=8, seeds=[1,2,3], variants=["equal","balanced"], precision="bf16")
print("RECIPE", freeze_recipe(Path("research/paired-severity-v1/recipe.json"), sources, config), flush=True)
PY
}
seal
for seed in 1 2 3; do
  for variant in equal balanced; do
    seal
    out="research/paired-severity-v1/${variant}-s${seed}"
    if test -e "$out"; then echo "Immutable run exists: $out"; exit 2; fi
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
      if test "$free" -ge 22000; then break; fi
      sleep 30
    done
    .runtime/bin/python scripts/paired_severity_train.py --data-root data-store \
      --manifest data-store/severity-pairs/manifest.json --out "$out" --seed "$seed" --variant "$variant"
  done
done
.runtime/bin/python scripts/score_paired_severity.py --data-root data-store \
  --manifest data-store/severity-pairs/manifest.json --runs research/paired-severity-v1 \
  --reference-root /home/mc/fires/research --out research/paired-severity-v1-result.json
