#!/bin/bash
# SPEC-74/75/73: Prithvi на CPU по мере готовности наборов; итоги после обучения C1-F.
cd $HOME/fires; export CUDA_VISIBLE_DEVICES=
while systemctl --user is-active -q fresh2-build; do sleep 60; done
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from src.comp.hls_eval import *; m,X,Y,V,_=load_dir('external/hls_fresh2'); prithvi_probs(X,'research/hls-fresh2-v1/prithvi.npy')" > logs/s74_prithvi.log 2>&1
while systemctl --user is-active -q floga-build; do sleep 60; done
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from src.comp.hls_eval import *; m,X,Y,V,_=load_dir('external/floga_hls'); prithvi_probs(X,'research/floga-hls-v1/prithvi.npy')" > logs/s73_prithvi.log 2>&1
until grep -q "c1f done" logs/c1f_queue.log 2>/dev/null; do sleep 60; done
unset CUDA_VISIBLE_DEVICES
.venv/bin/python scripts/exp_fresh_mix.py > logs/s7475_result.txt 2>&1
.venv/bin/python scripts/exp_mountains.py > logs/s73_result.txt 2>&1
echo "s737475 done $(date +%H:%M)" >> logs/c1f_queue.log
