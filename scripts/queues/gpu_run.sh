#!/bin/bash
# gpu_run.sh NEED_MB UNIT LOG [-E VAR=..]... -- cmd...  — общий старт задач на GPU (SPEC-69/70).
# ponytail: старт под одной блокировкой + пауза на разгон; рост памяти уже идущей задачи не ловит — тогда спасает повтор.
need=$1; unit=$2; log=$3; shift 3; envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
cd $HOME/fires
for try in 1 2 3; do
  (
    flock 9
    until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -ge $need ]; do sleep 60; done
    systemctl --user reset-failed $unit 2>/dev/null
    systemd-run --user --quiet --slice=fires.slice --unit=$unit -p MemoryMax=26G -p StandardOutput=file:$log -p StandardError=inherit -p WorkingDirectory=$HOME/fires -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${envs[@]}" "$@"
    sleep 180
  ) 9>/tmp/gpu_start.lock
  while systemctl --user is-active -q $unit; do sleep 30; done
  [ "$(systemctl --user show -p Result --value $unit)" = success ] && exit 0
  echo "$unit попытка $try упала $(date +%H:%M)" >> logs/gpu_retry.log; sleep 120
done
exit 1
