#!/bin/bash
# Run the streamed-network GPU phase on the shared A100 (queues on ~/.gpu.lock; never kills other holders).
# env: N (digits per network, default 10), NETS (comma list of residual,deep,cnn), LOG (log file, default stream.log)
cd ~/qwentensor_stream || exit 1
export QWENTPU_SRC=/home/ubuntu/qwentensor_run/qwentpu_src
export N=${N:-10}
LOG=${LOG:-stream.log}
echo "queued $(date -u +%FT%TZ)" > "$LOG"
flock -w 21600 ~/.gpu.lock ~/qwentpu_gpu/venv/bin/python scripts/gpu_run.py stream >> "$LOG" 2>&1
echo "finished $(date -u +%FT%TZ) rc=$?" >> "$LOG"
