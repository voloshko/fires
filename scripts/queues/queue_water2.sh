#!/bin/bash
# SPEC-78: горный набор-4 → слои → горный отбор (до замера) → замер масок воды и повтор C-коррекции.
cd $HOME/fires; export GDAL_CACHEMAX=512
[ -f external/hls_fresh4/manifest.json ] || FRESH_OUT=external/hls_fresh4 FRESH_SKIP=640 FRESH_N=400 FRESH_EXCLUDE=external/hls_fresh3/manifest.json \
  .venv/bin/python scripts/build_hls_fresh.py >> logs/fresh4_build.log 2>> logs/fresh4_build.err || exit 1
.venv/bin/python scripts/enrich_windows.py external/hls_fresh4 --mount >> logs/enrich_fresh4.log 2>> logs/enrich_fresh4.err || exit 1
.venv/bin/python scripts/exp_water2.py > logs/water2_result.txt 2> logs/water2.err || exit 1
echo "water2 done $(date +%H:%M)" >> logs/water2_queue.log
