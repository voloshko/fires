#!/bin/bash
# SPEC-83 ч. А: двойники L2A → замер переноса C1-MM.
cd $HOME/fires; export GDAL_CACHEMAX=512
.venv/bin/python scripts/build_s2_twins.py >> logs/s2_twins.log 2>> logs/s2_twins.err || exit 1
.venv/bin/python scripts/exp_s2_transfer.py > logs/s2_transfer.txt 2> logs/s2_transfer.err || exit 1
echo "s2 done $(date +%H:%M)" >> logs/s2_queue.log
