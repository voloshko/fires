#!/bin/bash
# SPEC-70: радарный сиам + ROT90=1 --gain 0.1, парно к v22.
cd $HOME/fires; G=/tmp/gpu_run.sh
for f in 0 1 2 3 4; do out=research/bs-confirm-siam-f$f-rot-v1; [ -f $out/summary.json ] && continue
  $G 14000 run-rot-f$f $HOME/fires/logs/rot_f$f.log -E SAR_GATE=1 -E ROT90=1 -- .venv/bin/python scripts/hypothesis_bs.py train --fold $f --seed $((20260930+f)) --variant siam --precision bf16 --oversample-faint 0.17 3 --gain 0.1 --boost $HOME/fires-hypotheses/research/bs-confirm-boost-f$f-v1 --out $out
  [ -f $out/summary.json ] && echo "rot fold $f done $(date +%H:%M)" >> logs/rot_queue.log; done
for s in 20260918 20260919; do out=research/bs-siam-rot-screen-$s; [ -f $out/summary.json ] && continue
  $G 14000 run-rotscr-$s $HOME/fires/logs/rot_screen_$s.log -E SAR_GATE=1 -E ROT90=1 -- .venv/bin/python scripts/hypothesis_bs.py train --seed $s --variant siam --precision bf16 --oversample-faint 0.17 3 --gain 0.1 --boost $HOME/fires-hypotheses/research/bs-boost-v1 --out $out
  [ -f $out/summary.json ] && echo "rot screen $s done $(date +%H:%M)" >> logs/rot_queue.log; done
.venv/bin/python scripts/exp_siam_rot.py > logs/rot_result.txt 2>&1
echo "rot done $(date +%H:%M)" >> logs/rot_queue.log
