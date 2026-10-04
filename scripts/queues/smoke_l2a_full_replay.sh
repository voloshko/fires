#!/usr/bin/env bash
# SPEC-101: actual odd/even/missing-twin and last-batch plumbing, not accuracy.
set -euo pipefail
cd /home/mc/fires-accuracy-next
while systemctl --user is-active --quiet fires-accuracy-full-replay-prepare-v2; do sleep 10; done
test -f data-store/l2a-full-replay-v1/manifest-v2.json
exec 9>research/accuracy-gpu.lock
flock 9
for variant in control replay; do
  .runtime/bin/python scripts/l2a_full_replay_train.py --data-root data-store \
    --manifest data-store/l2a-full-replay-v1/manifest-v2.json \
    --checkpoint /home/mc/fires/research/hls-c1mm-final-s1/model.pt \
    --seed 1 --variant "$variant" --smoke --out "research/l2a-full-replay-smoke-${variant}-s1"
done
