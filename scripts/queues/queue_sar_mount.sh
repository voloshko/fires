#!/bin/bash
# SPEC-85: радарные слои обучения → S1-M (3 сида) → первый пролёт и его радар на проверке → замер.
cd $HOME/fires; export GDAL_CACHEMAX=512; G=/tmp/gpu_run.sh; P=.venv/bin/python
$P scripts/build_s1_layers.py external/hls_fresh manifest.json >> logs/s1_layers.log 2>> logs/s1_layers.err || exit 1
$P scripts/build_s1_layers.py external/ems_hls manifest_train.json >> logs/s1_layers.log 2>> logs/s1_layers.err || exit 1
$P scripts/build_s1_layers.py external/floga_hls manifest_train.json >> logs/s1_layers.log 2>> logs/s1_layers.err || exit 1
for s in 1 2 3; do out=research/s1m-final-s$s
  [ -f $out/model.pt ] || $G 9000 run-s1m-s$s $HOME/fires/logs/s1m-s$s.log -- $P scripts/hls_train.py --fit all --config C1 --seed $s --channels s1 --extra-manifest external/hls_fresh/manifest_s1.json:1 --extra-manifest external/ems_hls/manifest_train_s1.json:2 --extra-manifest external/floga_hls/manifest_train_s1.json:2 --out $out
  [ -f $out/model.pt ] && echo "s1m s$s done $(date +%H:%M)" >> logs/sar_mount_queue.log || exit 1
done
for d in external/ems_hls external/floga_hls; do
  $P scripts/build_first_pass.py $d manifest_test.json >> logs/fp.log 2>> logs/fp.err || exit 1
  $P scripts/build_s1_layers.py $d manifest_test.json --first-pass >> logs/s1_layers.log 2>> logs/s1_layers.err || exit 1
done
$P scripts/exp_sar_mount.py > logs/sar_mount_result.txt 2> logs/sar_mount.err || exit 1
echo "sar_mount done $(date +%H:%M)" >> logs/sar_mount_queue.log
