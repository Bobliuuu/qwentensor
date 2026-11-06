#!/bin/bash
# Run on the A100 box: each job holds ~/.gpu.lock on its own. logs -> ~/qwentensor_b/logs/
cd ~/qwentensor_b; . ~/qwentpu_gpu/venv/bin/activate
export QWENTPU_SRC=~/qwentensor_run/qwentpu_src PYTHONPATH=~/qwentensor_run/qwentpu_src:~/qwentensor_b
mkdir -p logs
job() { # name model_dir tag phase
  echo "start $1 $(date)" >> logs/batch.status
  QWEN_DIR=$2 TAG=$3 flock -w 21600 ~/.gpu.lock python scripts/gpu_run.py $4 > logs/$1.log 2>&1
  echo "done $1 rc=$? $(date)" >> logs/batch.status
}
job residual_q08 ~/qwentpu_gpu/q08 0.8B residual
job throughput_q08 ~/qwentpu_gpu/q08 0.8B throughput
job digits_q2 ~/qwentpu_gpu/q2 2B digits
job general_q2 ~/qwentpu_gpu/q2 2B general
job digits_q4 ~/qwentpu_gpu/q4 4B digits
job general_q4 ~/qwentpu_gpu/q4 4B general
echo ALLDONE >> logs/batch.status
