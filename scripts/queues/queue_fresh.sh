#!/bin/bash
# SPEC-72: Prithvi на свежем наборе (CPU) сразу после сборки; итог — после переобучения C1.
cd $HOME/fires
while systemctl --user is-active -q hls-fresh-build; do sleep 60; done
[ -f external/hls_fresh/manifest.json ] || exit 1
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/exp_hls_fresh.py prithvi > logs/hls_fresh_prithvi.log 2>&1
until grep -q "c1 done" logs/c1_queue.log 2>/dev/null; do sleep 60; done
.venv/bin/python scripts/exp_hls_fresh.py > logs/hls_fresh_result.txt 2>&1
echo "fresh done $(date +%H:%M)" >> logs/c1_queue.log
