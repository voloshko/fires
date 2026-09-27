#!/bin/bash
# SPEC-84: EFFIS против экспертной разметки; карты пяти крупнейших пожаров Кипра 2025 с площадью EFFIS.
cd $HOME/fires; export GDAL_CACHEMAX=512
.venv/bin/python scripts/exp_effis.py > logs/effis_result.txt 2> logs/effis.err || exit 1
for r in 0 1 2 3 4; do
  MODEL=c1mm EVENT_RANK=$r .venv/bin/python scripts/cyprus_fire_map.py >> logs/effis_cyprus.log 2>> logs/effis_cyprus.err || echo "ранг $r: ошибка" >> logs/effis_cyprus.log
done
echo "effis done $(date +%H:%M)" >> logs/effis_queue.log
