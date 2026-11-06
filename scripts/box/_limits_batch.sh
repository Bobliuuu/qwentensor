#!/bin/bash
# Run on the A100 box: largest-machine probes, each job holds ~/.gpu.lock on its own. logs -> ~/qwentensor_limits/logs/
cd ~/qwentensor_limits; . ~/qwentpu_gpu/venv/bin/activate
export QWENTPU_SRC=~/qwentensor_run/qwentpu_src PYTHONPATH=~/qwentensor_run/qwentpu_src:~/qwentensor_limits
mkdir -p logs
job() { # name model_dir tag case
  echo "start $1 $(date)" >> logs/batch.status
  QWEN_DIR=$2 TAG=$3 flock -w 21600 ~/.gpu.lock python scripts/probe_limits_gpu.py $4 > logs/$1.log 2>&1
  echo "done $1 rc=$? $(date)" >> logs/batch.status
}
job wide126_q08 ~/qwentpu_gpu/q08 0.8B wide126
job deep126_q08 ~/qwentpu_gpu/q08 0.8B deep126
job rows32_q08 ~/qwentpu_gpu/q08 0.8B rows32
job wide126_q2 ~/qwentpu_gpu/q2 2B wide126
job wide126_q4 ~/qwentpu_gpu/q4 4B wide126
echo ALLDONE >> logs/batch.status
