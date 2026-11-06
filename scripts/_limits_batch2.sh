#!/bin/bash
# Run on the A100 box: second batch of largest-machine probes (63-input neurons, rows 32); each job holds ~/.gpu.lock on its own.
cd ~/qwentensor_limits; . ~/qwentpu_gpu/venv/bin/activate
export QWENTPU_SRC=~/qwentensor_run/qwentpu_src PYTHONPATH=~/qwentensor_run/qwentpu_src:~/qwentensor_limits
mkdir -p logs
job() { # name model_dir tag case
  echo "start $1 $(date)" >> logs/batch2.status
  QWEN_DIR=$2 TAG=$3 flock -w 21600 ~/.gpu.lock python scripts/probe_limits_gpu.py $4 > logs/$1.log 2>&1
  echo "done $1 rc=$? $(date)" >> logs/batch2.status
}
job wide63_q08 ~/qwentpu_gpu/q08 0.8B wide63
job rows32_q08 ~/qwentpu_gpu/q08 0.8B rows32
echo ALLDONE >> logs/batch2.status
