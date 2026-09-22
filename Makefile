PY := .venv/bin/python
CFG := configs/baseline_lgb.yaml

.PHONY: venv data baseline test clean-data kronos-pilot kronos-eval

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

# Stage 2: Kronos zero-shot signal on the point-in-time universe (clean window starts 2024-07, after Kronos's data cutoff)
KR_OUT := data/kronos/pilot_post2024h2.parquet
kronos-pilot:
	$(PY) -u scripts/03_kronos_pilot.py generate --start 2024-07-01 --end 2026-09-21 --every 5 --lookback 90 --pred-len 5 --samples 16 --batch 96 --size small --out $(KR_OUT)

kronos-eval:
	$(PY) -u scripts/03_kronos_pilot.py evaluate --features $(KR_OUT) --pred-len 5 --tag post2024h2
