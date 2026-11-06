#!/bin/bash
# Run on the A100 box: each job holds ~/.gpu.lock on its own. logs -> ~/qwentensor_wide/logs/
cd ~/qwentensor_wide; . ~/qwentpu_gpu/venv/bin/activate
export QWENTPU_SRC=~/qwentensor_run/qwentpu_src PYTHONPATH=~/qwentensor_run/qwentpu_src:~/qwentensor_wide
mkdir -p logs
job() { # name nets n
  echo "start $1 $(date)" >> logs/batch.status
  QWEN_DIR=$HOME/qwentpu_gpu/q08 TAG=0.8B NETS=$2 N=$3 flock -w 43200 ~/.gpu.lock python scripts/gpu_run.py wide > logs/$1.log 2>&1
  echo "done $1 rc=$? $(date)" >> logs/batch.status
}
job wide_residual residual 10
job wide_deep deep 10
job wide_cnn cnn 6
echo ALLDONE >> logs/batch.status
