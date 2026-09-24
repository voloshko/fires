#!/bin/bash
# SPEC-81: чистый резерв EMS → C1-M (HLS 540 + свежий MTBS 300 + резерв EMS ×3), сиды 1–5 по одному (память хоста) → замер.
cd $HOME/fires; G=/tmp/gpu_run.sh
[ -f external/ems_hls/manifest_train.json ] || .venv/bin/python scripts/ems_reserve_clean.py >> logs/c1m_queue.log 2>&1 || exit 1
for s in 1 2 3 4 5; do out=research/hls-c1m-final-s$s
  [ -f $out/model.pt ] || $G 12000 run-c1m-s$s $HOME/fires/logs/hls-c1m-s$s.log -- .venv/bin/python scripts/hls_train.py --fit all --config C1 --seed $s --extra external/hls_fresh --extra-manifest external/ems_hls/manifest_train.json:3 --out $out
  [ -f $out/model.pt ] && echo "c1m s$s done $(date +%H:%M)" >> logs/c1m_queue.log || exit 1
done
.venv/bin/python scripts/exp_c1m.py > logs/c1m_result.txt 2> logs/c1m.err || exit 1
echo "c1m done $(date +%H:%M)" >> logs/c1m_queue.log
