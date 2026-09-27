#!/bin/bash
# SPEC-89: снимки «до» для окон с тяжестью в свежих 3 и 4 → гибрид против dNBR.
cd $HOME/fires; export GDAL_CACHEMAX=512; P=.venv/bin/python
for d in external/hls_fresh3 external/hls_fresh4; do $P scripts/build_pre_hls.py $d manifest_sev.json >> logs/sev_hybrid.log 2>> logs/sev_hybrid.err || exit 1; done
$P scripts/exp_sev_hybrid.py > logs/sev_hybrid_result.txt 2> logs/sev_hybrid_eval.err || exit 1
echo "sev_hybrid done $(date +%H:%M)" >> logs/sev_hybrid_queue.log
