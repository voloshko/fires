#!/bin/bash
# SPEC-72: переобучение C1 (сиды 1–5, все 540 сцен) с сохранением весов; два одновременно.
cd $HOME/fires; G=/tmp/gpu_run.sh
worker() { for s in 1 2 3 4 5; do out=research/hls-c1-final-s$s; mkdir research/.lock-c1-s$s 2>/dev/null || continue
  [ -f $out/model.pt ] || $G 9500 run-c1-s$s $HOME/fires/logs/hls-c1-s$s.log -- .venv/bin/python scripts/hls_train.py --fit all --config C1 --seed $s --out $out
  [ -f $out/summary.json ] && echo "c1 s$s done $(date +%H:%M)" >> logs/c1_queue.log; done; }
worker & sleep 30; worker & wait
echo "c1 done $(date +%H:%M)" >> logs/c1_queue.log
