.PHONY: install dev test test-gpu site serve
install:
	pip install -e ".[dev]"
# QWENTPU_PATH=~/qwentpu_src/qwentpu make dev
dev:
	@test -n "$(QWENTPU_PATH)" || (echo "set QWENTPU_PATH to a qwentpu checkout"; exit 1)
	pip install -e "$(QWENTPU_PATH)"
	pip install -e . --no-deps && pip install pytest
test:
	pytest -q -m "not gpu"
test-gpu:
	pytest -q -m gpu
site:
	sh docs/site/sync.sh
serve:
	cd docs && python -m http.server 8000
