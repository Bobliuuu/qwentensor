.PHONY: install dev test site serve
install:
	pip install -e ".[dev]"
# QWENTPU_PATH=/path/to/a/qwentpu/checkout make dev   (develop against a local qwenTPU instead of the one pip installs)
dev:
	@test -n "$(QWENTPU_PATH)" || (echo "set QWENTPU_PATH to a qwentpu checkout"; exit 1)
	pip install -e "$(QWENTPU_PATH)"
	pip install -e . --no-deps && pip install pytest scikit-learn
# CPU tests (what CI runs). Real-model runs are scripts: scripts/gpu_run.py, scripts/gpu_vec.py (need QWEN_DIR and CUDA).
test:
	pytest -q
site:
	sh docs/site/sync.sh
serve:
	cd docs && python -m http.server 8000
