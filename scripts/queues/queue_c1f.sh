#!/bin/bash
# SPEC-75: C1 на HLS 540 + первом свежем наборе 300, сиды 1–5, по два одновременно.
cd $HOME/fires; G=/tmp/gpu_run.sh
worker() { for s in 1 2 3 4 5; do out=research/hls-c1f-final-s$s; mkdir research/.lock-c1f-s$s 2>/dev/null || continue
  [ -f $out/model.pt ] || $G 9500 run-c1f-s$s $HOME/fires/logs/hls-c1f-s$s.log -- .venv/bin/python scripts/hls_train.py --fit all --config C1 --seed $s --extra external/hls_fresh --out $out
  [ -f $out/summary.json ] && echo "c1f s$s done $(date +%H:%M)" >> logs/c1f_queue.log; done; }
worker & sleep 30; worker & wait
echo "c1f done $(date +%H:%M)" >> logs/c1f_queue.log
