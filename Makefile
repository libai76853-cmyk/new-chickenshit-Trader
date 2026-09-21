PY := .venv/bin/python
CFG := configs/baseline_lgb.yaml

.PHONY: venv data baseline test clean-data

venv:
	/opt/homebrew/bin/python3.12 -m venv .venv
	.venv/bin/pip install -U pip wheel setuptools
	.venv/bin/pip install -r requirements.txt

data:
	$(PY) -u scripts/01_build_data.py --config $(CFG)

baseline:
	$(PY) -u scripts/02_run_baseline.py --config $(CFG)

test:
	$(PY) -m pytest -q tests

clean-data:
	rm -rf data/raw data/normalized data/qlib data/artifacts mlruns
