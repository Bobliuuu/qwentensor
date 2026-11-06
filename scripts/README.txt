make test       CPU tests (what CI runs)
make site       regenerate docs/site/data/results.js after editing results.json
make serve      serve docs/ on :8000 (the page also works from file://)

Real-model runs (need a Qwen3.5 checkpoint in QWEN_DIR and CUDA): gpu_run.py, gpu_vec.py, probe_limits_gpu.py.
box/ holds the scripts that produced the committed A100 logs; see box/README.txt.
