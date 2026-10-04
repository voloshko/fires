#!/usr/bin/env bash
# SPEC-116: no fresh source I/O until full114 scoring and all four guards pass.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-equal-cost-additions-score-v1; do sleep 30; done
if .runtime/bin/python - <<'PY'
import json
from pathlib import Path
from src.comp.equal_cost_eastern import require_eligible
p=json.loads(Path('research/equal-cost-additions-v1/development.json').read_text())
if p.get('eligible') is False:
    print('SKIPPED_INELIGIBLE_SPEC114', json.dumps(p.get('guards')), flush=True)
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
.runtime/bin/python scripts/collect_equal_cost_eastern.py --data-root data-store \
  --selection data-store/eastern-holdout-plan-v1/selection.json \
  --candidate research/equal-cost-additions-v1/development.json \
  --runs research/equal-cost-additions-v1 --reference-root /home/mc/fires/research \
  --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars \
  --out data-store/eastern-confirmation-v2
.runtime/bin/python - <<'PY'
import json
from pathlib import Path
from src.comp.equal_cost_eastern import require_admission
require_admission(json.loads(Path('data-store/eastern-confirmation-v2/manifest.json').read_text()))
PY
systemd-run --user --unit=fires-accuracy-equal-cost-eastern-score-v1 \
  --property=WorkingDirectory=/home/mc/fires-accuracy-next \
  --property=CPUQuota=400% --property=MemoryMax=12G \
  --setenv=OMP_NUM_THREADS=4 --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/bash /home/mc/fires-accuracy-next/scripts/queues/queue_score_equal_cost_eastern.sh
