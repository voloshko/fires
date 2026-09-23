#!/bin/bash
# SPEC-76/77: горный свежий набор → слои → отбор горных окон (до замера) → замер.
cd $HOME/fires; export GDAL_CACHEMAX=512
while systemctl --user is-active -q fresh3-build; do sleep 60; done
[ -f external/hls_fresh3/manifest.json ] || exit 1
.venv/bin/python scripts/enrich_windows.py external/hls_fresh3 --mount >> logs/enrich_fresh3.log 2>> logs/enrich_fresh3.err || exit 1
while systemctl --user is-active -q enrich-floga; do sleep 60; done
.venv/bin/python scripts/exp_terrain.py > logs/terrain_result.txt 2> logs/terrain.err
echo "terrain done $(date +%H:%M)" >> logs/terrain_queue.log
