#!/bin/bash
# SPEC-73: сборка гор пачками по 20 событий (каждая — отдельный процесс), манифест, Prithvi на CPU, замер.
cd $HOME/fires; export GDAL_CACHEMAX=512
for a in $(seq 0 20 190); do
  [ -f logs/floga_part_$a.done ] && continue
  FLOGA_RANGE=$a:$((a+20)) .venv/bin/python scripts/build_floga_hls.py >> logs/floga_build.log 2>> logs/floga_build.err && touch logs/floga_part_$a.done
done
FLOGA_ASSEMBLE=1 .venv/bin/python scripts/build_floga_hls.py >> logs/floga_build.log 2>> logs/floga_build.err
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/exp_mountains.py > logs/s73_result.txt 2>&1
echo "floga done $(date +%H:%M)" >> logs/floga_queue.log
