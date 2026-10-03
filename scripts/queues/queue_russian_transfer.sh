#!/usr/bin/env bash
# SPEC-92: matched transfer after the earlier GPU campaign, with a sealed recipe.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-severity-train-v1 || \
      systemctl --user is-active --quiet fires-accuracy-russian-prepare-v1; do sleep 30; done
test -f data-store/russia/manifest.json
.runtime/bin/python -c 'import json; x=json.load(open("data-store/russia/manifest.json")); assert x["ready_for_training"] and not x["qc_probe"]'
exec 9>research/accuracy-gpu.lock
flock 9
seal() {
  .runtime/bin/python - <<'PY'
from src.comp.accuracy_artifacts import execution_sources, freeze_recipe
from src.comp.accuracy_contract import sha256_file
sources = execution_sources("scripts/russian_transfer_train.py", "src/comp/russian_transfer.py",
                            "scripts/forest_pair_train.py", "scripts/prepare_russian_windows.py",
                            "scripts/queues/queue_russian_transfer.sh")
config = dict(base=sha256_file("data-store/forest-pairs/manifest.json"),
              russian=sha256_file("data-store/russia/manifest.json"), epochs=30, lr=3e-5,
              batch=8, seeds=[1,2,3], variants=["control","reserve"],
              initializers=[sha256_file(f"research/forest-pair-v1/pair-s{s}/model.pt") for s in (1,2,3)])
print("RECIPE", freeze_recipe("research/russian-transfer-v1/recipe.json", sources, config), flush=True)
PY
}
seal
for seed in 1 2 3; do
  for variant in control reserve; do
    seal
    out="research/russian-transfer-v1/${variant}-s${seed}"
    if test -e "$out"; then echo "Immutable run exists: $out"; exit 2; fi
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
      if test "$free" -ge 22000; then break; fi
      sleep 30
    done
    .runtime/bin/python scripts/russian_transfer_train.py --data-root data-store \
      --base-manifest data-store/forest-pairs/manifest.json --russian-manifest data-store/russia/manifest.json \
      --checkpoint "research/forest-pair-v1/pair-s${seed}/model.pt" --seed "$seed" --variant "$variant" --out "$out"
  done
done
.runtime/bin/python scripts/score_russian_transfer.py --data-root data-store \
  --base-manifest data-store/forest-pairs/manifest.json --russian-manifest data-store/russia/manifest.json \
  --runs research/russian-transfer-v1 --initializer-root research/forest-pair-v1 \
  --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars \
  --out research/russian-transfer-v1-result.json
