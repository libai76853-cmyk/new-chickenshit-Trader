"""Build the Qlib dataset: universe -> Yahoo raw -> normalized CSV -> Qlib bin.

Usage: .venv/bin/python scripts/01_build_data.py [--config configs/baseline_lgb.yaml] [--refresh-universe] [--redownload]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import yaml
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ljs.universe import load_universe, BENCHMARK  # noqa: E402
from ljs.data_yahoo import download_raw, load_calendar, normalize_all  # noqa: E402
from ljs.dump_qlib import dump_all, write_universe_instruments, write_future_calendar, write_members_asof_instruments  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "baseline_lgb.yaml"))
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--redownload", action="store_true", help="ignore cached raw CSVs")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N symbols")
    args = ap.parse_args()

    full = yaml.safe_load(Path(args.config).read_text())
    cfg = full["data"]
    start = cfg["start"]
    end = cfg["end"] or (dt.date.today() + dt.timedelta(days=1)).isoformat()  # yfinance end is exclusive
    raw_dir, norm_dir, qlib_dir = (REPO / cfg[k] for k in ("raw_dir", "norm_dir", "qlib_dir"))
    bench = cfg.get("benchmark", BENCHMARK)

    uni = load_universe(REPO / cfg["universe_cache"], refresh=args.refresh_universe)
    symbols = uni["yahoo"].tolist()
    if args.limit:
        symbols = symbols[: args.limit]
    logger.info(f"universe: {len(symbols)} symbols + benchmark {bench}; range {start} -> {end}")

    download_raw(symbols + [bench], start, end, raw_dir, chunk_size=cfg["chunk_size"], skip_existing=not args.redownload)
    calendar = load_calendar(raw_dir / f"{bench}.csv")
    logger.info(f"calendar from {bench}: {len(calendar)} trading days {calendar.min().date()} -> {calendar.max().date()}")
    kept = normalize_all(raw_dir, norm_dir, calendar, min_rows=cfg["min_rows"])
    dump_all(norm_dir, qlib_dir, max_workers=cfg["dump_workers"])
    n = write_universe_instruments(qlib_dir, cfg["universe_name"], exclude={bench})
    write_future_calendar(qlib_dir)
    asof = full["dataset"]["segments"]["test"][0]
    write_members_asof_instruments(qlib_dir, uni, asof, name=f"{cfg['universe_name']}_pre{asof[:4]}")
    logger.info(f"DONE: {n} tradable symbols in instruments/{cfg['universe_name']}.txt; qlib_dir={qlib_dir}")


if __name__ == "__main__":
    main()
