#!/usr/bin/env bash
# SPEC-119: no fresh source I/O until full118 scoring and all four guards pass.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-five-seed-additions-score-v1; do sleep 30; done
if .runtime/bin/python - <<'PY'
import json
from pathlib import Path
from src.comp.five_seed_eastern import require_eligible
p=json.loads(Path('research/five-seed-additions-v1/development.json').read_text())
if p.get('eligible') is False:
    print('SKIPPED_INELIGIBLE_SPEC118', json.dumps(p.get('guards')), flush=True)
    raise SystemExit(42)
require_eligible(p)
PY
then
  candidate_status=0
else
  candidate_status=$?
fi
if test "$candidate_status" -eq 42; then exit 0; fi
if test "$candidate_status" -ne 0; then exit "$candidate_status"; fi
.runtime/bin/python scripts/collect_five_seed_eastern.py --data-root data-store \
  --selection data-store/eastern-holdout-plan-v1/selection.json \
  --candidate research/five-seed-additions-v1/development.json \
  --runs research/five-seed-additions-v1 --reference-root /home/mc/fires/research \
  --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars \
  --out data-store/eastern-confirmation-v3
.runtime/bin/python - <<'PY'
import json
from pathlib import Path
from src.comp.five_seed_eastern import require_admission
require_admission(json.loads(Path('data-store/eastern-confirmation-v3/manifest.json').read_text()))
PY
systemd-run --user --unit=fires-accuracy-five-seed-eastern-score-v1 \
  --property=WorkingDirectory=/home/mc/fires-accuracy-next \
  --property=CPUQuota=400% --property=MemoryMax=12G \
  --setenv=OMP_NUM_THREADS=4 --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/bash /home/mc/fires-accuracy-next/scripts/queues/queue_score_five_seed_eastern.sh
