"""Build the Qlib dataset: universe (current + historical members) -> Yahoo raw -> normalized CSV -> Qlib bin
-> instruments files: <universe> (current members), <universe>_pre<year> (members as of test start),
<universe>_pit (point-in-time spells from the Wikipedia change table).

Usage: .venv/bin/python scripts/01_build_data.py [--config configs/baseline_lgb.yaml] [--refresh-universe] [--redownload] [--limit N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ljs.universe import load_universe, BENCHMARK  # noqa: E402
from ljs.data_yahoo import download_raw, load_calendar, normalize_all  # noqa: E402
from ljs.dump_qlib import dump_all, write_universe_instruments, write_future_calendar, write_members_asof_instruments  # noqa: E402
from ljs.membership import load_changes, build_spells, data_ranges, filter_spells_by_data, write_pit_instruments, coverage_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "baseline_lgb.yaml"))
    ap.add_argument("--refresh-universe", action="store_true", help="re-fetch Wikipedia constituents + change table")
    ap.add_argument("--redownload", action="store_true", help="ignore cached raw CSVs and the no-data list")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N current symbols")
    args = ap.parse_args()

    full = yaml.safe_load(Path(args.config).read_text())
    cfg = full["data"]
    start = cfg["start"]
    today = dt.date.today()
    end = cfg["end"] or (today + dt.timedelta(days=1)).isoformat()  # yfinance end is exclusive
    raw_dir, norm_dir, qlib_dir = (REPO / cfg[k] for k in ("raw_dir", "norm_dir", "qlib_dir"))
    bench = cfg.get("benchmark", BENCHMARK)
    uname = cfg["universe_name"]

    # ---- universe: current members + point-in-time spells
    uni = load_universe(REPO / cfg["universe_cache"], refresh=args.refresh_universe)
    changes = load_changes(REPO / cfg["changes_cache"], refresh=args.refresh_universe)
    current = uni["yahoo"].tolist()
    if args.limit:
        current = current[: args.limit]
    spells, diag = build_spells(uni, changes, start=start, end=today.isoformat())
    hist_syms = sorted(set(spells["symbol"]) - set(uni["yahoo"]))
    symbols = current + ([] if args.limit else hist_syms)
    logger.info(f"universe: {len(current)} current + {len(hist_syms)} historical symbols + benchmark {bench}; range {start} -> {end}")
    logger.info("membership diagnostics: " + ", ".join(f"{k}={len(v)}" for k, v in diag.items()))

    # ---- data
    download_raw(symbols + [bench], start, end, raw_dir, chunk_size=cfg["chunk_size"],
                 skip_existing=not args.redownload, no_data_path=None if args.redownload else raw_dir / "_no_data.txt")
    calendar = load_calendar(raw_dir / f"{bench}.csv")
    logger.info(f"calendar from {bench}: {len(calendar)} trading days {calendar.min().date()} -> {calendar.max().date()}")
    normalize_all(raw_dir, norm_dir, calendar, min_rows=cfg["min_rows"])
    dump_all(norm_dir, qlib_dir, max_workers=cfg["dump_workers"])

    # ---- instruments files
    n_cur = write_universe_instruments(qlib_dir, uname, exclude={bench, *hist_syms})
    write_future_calendar(qlib_dir)
    asof = full["dataset"]["segments"]["test"][0]
    write_members_asof_instruments(qlib_dir, uni, asof, name=f"{uname}_pre{asof[:4]}")
    ranges = data_ranges(norm_dir)
    spells_f = filter_spells_by_data(spells, ranges)
    write_pit_instruments(qlib_dir, spells_f, name=f"{uname}_pit")
    out = spells_f.copy()
    for c in ("start", "end", "start_eff", "end_eff"):
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    out.to_csv(REPO / "data" / "membership_spells.csv", index=False)
    cov = coverage_summary(spells_f, changes, set(uni["yahoo"]), start, today.isoformat())
    cov["diagnostics"] = {k: len(v) for k, v in diag.items()}
    cov["renamed_or_missing_removal"] = diag["renamed_or_missing_removal"]
    (REPO / "data" / "membership_coverage.json").write_text(json.dumps(cov, indent=1, ensure_ascii=False))
    logger.info("PIT coverage: " + json.dumps({k: cov[k] for k in ("distinct_removed_tickers", "removed_tickers_with_data", "spells_total", "spells_kept", "member_days_total", "member_days_with_data")}))
    logger.info(f"removed by category: {cov['removed_by_category']}")
    logger.info(f"DONE: {n_cur} current symbols in instruments/{uname}.txt; PIT spells in instruments/{uname}_pit.txt; qlib_dir={qlib_dir}")


if __name__ == "__main__":
    main()
