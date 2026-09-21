"""Train + evaluate the Stage-1 baseline and write reports/baseline_lgb_<date>.md

Usage: .venv/bin/python scripts/02_run_baseline.py [--config configs/baseline_lgb.yaml] [--tag NAME]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "baseline_lgb.yaml"))
    ap.add_argument("--tag", default=None, help="suffix for report files (default: today)")
    args = ap.parse_args()
    from ljs.baseline_lgb import run

    path = run(args.config, tag=args.tag)
    print(f"REPORT {path}")


if __name__ == "__main__":
    main()
