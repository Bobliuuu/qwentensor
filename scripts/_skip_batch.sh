#!/bin/bash
# Run on the A100 box: each job holds ~/.gpu.lock on its own. usage: _skip_batch.sh <name> <phase> [args...]; logs -> ~/qwentensor_skip/logs/
cd ~/qwentensor_skip; . ~/qwentpu_gpu/venv/bin/activate
export QWENTPU_SRC=${QWENTPU_SRC:-~/qwentensor_run/qwentpu_src} PYTHONPATH=${QWENTPU_SRC:-~/qwentensor_run/qwentpu_src}:~/qwentensor_skip
mkdir -p logs
name=$1; shift
echo "start $name $(date)" >> logs/batch.status
QWEN_DIR=~/qwentpu_gpu/q08 TAG=0.8B flock -w 21600 ~/.gpu.lock python scripts/gpu_skip.py "$@" > logs/$name.log 2>&1
echo "done $name rc=$? $(date)" >> logs/batch.status
