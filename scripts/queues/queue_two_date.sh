#!/bin/bash
# SPEC-87: снимки «до» для проверки EMS и FLOGA → замер «модель или dNBR».
cd $HOME/fires; export GDAL_CACHEMAX=512; P=.venv/bin/python
for d in external/ems_hls external/floga_hls; do $P scripts/build_pre_hls.py $d manifest_test.json >> logs/two_date.log 2>> logs/two_date.err || exit 1; done
$P scripts/exp_two_date.py > logs/two_date_result.txt 2> logs/two_date_eval.err || exit 1
echo "two_date done $(date +%H:%M)" >> logs/two_date_queue.log
