#!/usr/bin/env bash
set -euo pipefail
cd /home/mc/fires-accuracy-next
exec 9>research/accuracy-gpu.lock
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if test "$free" -ge 22000; then break; fi
  sleep 30
done
.runtime/bin/python scripts/score_coupled_member_inference.py --data-root data-store \
  --members-root research/five-seed-additions-v1 --runs research/coupled-member-v1 \
  --reference-root /home/mc/fires/research \
  --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars \
  --out research/coupled-member-v1/development.json
