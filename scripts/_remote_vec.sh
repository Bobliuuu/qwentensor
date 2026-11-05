#!/bin/bash
# Run on the A100 box: each job holds ~/.gpu.lock on its own. logs -> ~/qwentensor_vec/logs/
cd ~/qwentensor_vec; . ~/qwentpu_gpu/venv/bin/activate
export QWENTPU_SRC=~/qwentensor_run/qwentpu_src PYTHONPATH=~/qwentensor_run/qwentpu_src:~/qwentensor_vec
mkdir -p logs
job() { # name phase variants...
  name=$1; shift
  echo "start $name $(date)" >> logs/batch.status
  QWEN_DIR=~/qwentpu_gpu/q08 TAG=0.8B flock -w 21600 ~/.gpu.lock python scripts/gpu_vec.py "$@" > logs/$name.log 2>&1
  echo "done $name rc=$? $(date)" >> logs/batch.status
}
job bench bench scalar vec1 vec2
job exact exact vec1 vec2
job general general vec1
echo ALLDONE >> logs/batch.status
