#!/usr/bin/env bash
# SPEC-109: score only after six completed full runs, never a partial ensemble.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-temporal-adapter-train-v1; do sleep 30; done
.runtime/bin/python - <<'PY'
import json
from pathlib import Path
for seed in (1,2,3):
    for variant in ('pair','post'):
        p=Path('research/temporal-adapter-v1')/f'{variant}-s{seed}'
        assert (p/'adapter.pt').exists()
        assert json.loads((p/'summary.json').read_text())['completed']
PY
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
.runtime/bin/python scripts/score_temporal_adapter.py --data-root data-store \
  --runs research/temporal-adapter-v1 --reference-root /home/mc/fires/research \
  --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars \
  --out research/temporal-adapter-v1/development.json
