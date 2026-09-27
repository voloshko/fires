#!/bin/bash
# SPEC-86: тяжесть MTBS → снимки «до» для проверки → C1-S (3 сида) на свежих 1, 3, 4 → замер на свежем 2.
cd $HOME/fires; export GDAL_CACHEMAX=512; G=/tmp/gpu_run.sh; P=.venv/bin/python
$P scripts/build_mtbs_sev.py external/hls_fresh external/hls_fresh2 external/hls_fresh3 external/hls_fresh4 >> logs/sev.log 2>> logs/sev.err || exit 1
$P scripts/build_pre_hls.py external/hls_fresh2 manifest.json >> logs/sev.log 2>> logs/sev.err || exit 1
for s in 1 2 3; do out=research/c1s-final-s$s
  [ -f $out/model.pt ] || $G 11000 run-c1s-s$s $HOME/fires/logs/c1s-s$s.log -- $P scripts/hls_train.py --fit all --config C1 --seed $s --labels sev --channels hls --extra-manifest external/hls_fresh/manifest_sev.json:1 --extra-manifest external/hls_fresh3/manifest_sev.json:1 --extra-manifest external/hls_fresh4/manifest_sev.json:1 --out $out
  [ -f $out/model.pt ] && echo "c1s s$s done $(date +%H:%M)" >> logs/sev_queue.log || exit 1
done
$P scripts/exp_severity.py > logs/sev_result.txt 2> logs/sev_eval.err || exit 1
echo "sev done $(date +%H:%M)" >> logs/sev_queue.log
