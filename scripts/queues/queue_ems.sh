#!/bin/bash
# SPEC-79: EMS → окна HLS → слои для проверочной половины → замер (V1 маска на побережье, V2 C-коррекция, строка стенда).
cd $HOME/fires; export GDAL_CACHEMAX=512
.venv/bin/python scripts/build_ems_hls.py >> logs/ems_build.log 2>> logs/ems_build.err || exit 1
.venv/bin/python scripts/enrich_windows.py external/ems_hls --manifest=manifest_test.json >> logs/enrich_ems.log 2>> logs/enrich_ems.err || exit 1
.venv/bin/python scripts/exp_water2.py external/ems_hls manifest_test.json research/ems-test-v1 > logs/ems_result.txt 2> logs/ems.err || exit 1
echo "ems done $(date +%H:%M)" >> logs/ems_queue.log
