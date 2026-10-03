#!/usr/bin/env bash
# SPEC-99: frozen inference only, serialized after Russian transfer and collection.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-russian-transfer-v1 || \
      systemctl --user is-active --quiet fires-accuracy-retrospective-prepare-v1; do sleep 30; done
seal() {
.runtime/bin/python - <<'PY'
import json
from src.comp.accuracy_artifacts import execution_sources, freeze_recipe
from src.comp.accuracy_contract import sha256_file
from src.comp.fresh_severity import verify_development_result
manifest="data-store/severity-retrospective-v1/manifest.json"
x=json.load(open(manifest))
assert x["ready_for_evaluation"] and not x["qc_probe"] and x["spec_id"]=="SPEC-99"
verify_development_result("research/paired-severity-v1-result.json")
sources=execution_sources("scripts/score_fresh_paired_severity.py","scripts/prepare_fresh_paired_severity.py",
    "scripts/score_paired_severity.py","scripts/score_forest_pair.py","scripts/forest_pair_train.py",
    "src/comp/fresh_severity.py","src/comp/paired_severity.py","src/comp/severity_layers.py",
    "src/comp/terrain.py","scripts/queues/queue_fresh_paired_severity.sh")
config=dict(manifest=sha256_file(manifest),profile="retrospective-2016-17",tta=8,precision="bf16",
    model_hashes=[sha256_file(f"research/paired-severity-v1/equal-s{s}/model.pt") for s in (1,2,3)],
    development=sha256_file("research/paired-severity-v1-result.json"))
print("RECIPE",freeze_recipe("research/fresh-retrospective-v1-recipe.json",sources,config),flush=True)
PY
}
seal
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
seal
.runtime/bin/python scripts/score_fresh_paired_severity.py --profile retrospective-2016-17 \
  --data-root data-store --manifest data-store/severity-retrospective-v1/manifest.json \
  --runs research/paired-severity-v1 --development-result research/paired-severity-v1-result.json \
  --reference-root /home/mc/fires/research --out research/fresh-retrospective-v1-result.json
