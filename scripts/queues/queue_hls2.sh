#!/bin/bash
# SPEC-69 финал: два сида одновременно (если помещаются), затем вердикт и разбор SPEC-71 на ансамбле.
cd $HOME/fires; G=/tmp/gpu_run.sh
until [ -f research/hls-beat-C2-inner/summary.json ]; do sleep 60; done
.venv/bin/python scripts/exp_hls_beat.py select > logs/hls_select.txt 2>&1 || exit 1
W=$(python3 -c "import json;print(json.load(open('research/hls-beat-select.json'))['winner'])"); echo "select $W $(date +%H:%M)" >> logs/hls_queue.log
NEED=$([ $W = C2 ] && echo 14000 || echo 11000)
worker() { for s in 1 2 3 4 5; do out=research/hls-beat-final-s$s; mkdir research/.lock-final-s$s 2>/dev/null || continue
  [ -f $out/summary.json ] || $G $NEED run-hls-final-s$s $HOME/fires/logs/hls-beat-final-s$s.log -- .venv/bin/python scripts/hls_train.py --fit all --config $W --seed $s --out $out
  [ -f $out/summary.json ] && echo "hls-beat-final-s$s done $(date +%H:%M)" >> logs/hls_queue.log; done; }
worker & sleep 30; worker & wait
.venv/bin/python scripts/exp_hls_beat.py final > logs/hls_final.txt 2>&1
T=$(python3 -c "import json;print(json.load(open('research/hls-beat-select.json'))['threshold'])")
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/exp_hls_fair.py research/hls-beat-final-ensemble.npy $T > logs/hls_fair_final.txt 2>&1
echo "hls done $(date +%H:%M)" >> logs/hls_queue.log
