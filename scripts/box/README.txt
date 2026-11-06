Scripts that were run on the project's own A100 box (provenance for results/a100_*.log and results/a100_limits_*.log).

They are kept as a record of how each log was produced, not as portable tooling: they `cd` into scratch copies of the repo under the home directory
(~/qwentensor_*), activate a venv at ~/qwentpu_gpu/venv, point QWEN_DIR at ~/qwentpu_gpu/q08 (0.8B), and queue every job on ~/.gpu.lock with flock.
To reproduce on another machine, run the underlying commands directly, with a Qwen3.5 checkpoint and CUDA:

    QWEN_DIR=/path/to/Qwen3.5-0.8B python scripts/gpu_run.py digits | general | residual | throughput | stream | wide | wide784
    QWEN_DIR=/path/to/Qwen3.5-0.8B python scripts/gpu_vec.py exact | general | bench
    QWEN_DIR=/path/to/Qwen3.5-0.8B python scripts/probe_limits_gpu.py wide126 | deep126 | rows32 | wide63

(add QWENTPU_SRC=/path/to/qwentpu to use a local qwenTPU checkout instead of the installed one.)
