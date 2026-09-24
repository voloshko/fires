#!/bin/bash
# SPEC-82: разбиение FLOGA → C1-MM (HLS 540 + MTBS 300 + EMS 49 ×2 + FLOGA-обучение ×2), сиды по одному → замер.
cd $HOME/fires; G=/tmp/gpu_run.sh
[ -f external/floga_hls/manifest_train.json ] || .venv/bin/python scripts/mixed_reserve.py >> logs/c1mm_queue.log 2>&1 || exit 1
for s in 1 2 3 4 5; do out=research/hls-c1mm-final-s$s
  [ -f $out/model.pt ] || $G 12000 run-c1mm-s$s $HOME/fires/logs/hls-c1mm-s$s.log -- .venv/bin/python scripts/hls_train.py --fit all --config C1 --seed $s --extra external/hls_fresh --extra-manifest external/ems_hls/manifest_train.json:2 --extra-manifest external/floga_hls/manifest_train.json:2 --out $out
  [ -f $out/model.pt ] && echo "c1mm s$s done $(date +%H:%M)" >> logs/c1mm_queue.log || exit 1
done
.venv/bin/python scripts/exp_c1mm.py > logs/c1mm_result.txt 2> logs/c1mm.err || exit 1
echo "c1mm done $(date +%H:%M)" >> logs/c1mm_queue.log
