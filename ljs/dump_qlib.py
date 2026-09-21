"""Convert normalized per-symbol CSVs into Qlib binary data using the vendored scripts/dump_bin.py,
then write a tradable-universe instruments file that excludes the benchmark."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
DUMP_BIN = REPO / "scripts" / "dump_bin.py"


def dump_all(norm_dir: Path, qlib_dir: Path, max_workers: int = 8) -> None:
    qlib_dir = Path(qlib_dir)
    if qlib_dir.exists():
        shutil.rmtree(qlib_dir)  # dump_all is a full rebuild
    cmd = [
        sys.executable, str(DUMP_BIN), "dump_all",
        "--data_path", str(norm_dir), "--qlib_dir", str(qlib_dir), "--freq", "day",
        "--max_workers", str(max_workers), "--exclude_fields", "symbol,date",
        "--symbol_field_name", "symbol", "--date_field_name", "date",
    ]
    logger.info("running dump_bin: " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(r.stderr[-3000:])
        raise RuntimeError("dump_bin failed")
    logger.info("dump_bin done")


def write_universe_instruments(qlib_dir: Path, name: str, exclude: set[str]) -> int:
    """Create instruments/<name>.txt = all.txt minus excluded symbols (e.g. the benchmark ETF)."""
    inst_dir = Path(qlib_dir) / "instruments"
    lines = (inst_dir / "all.txt").read_text().splitlines()
    keep = [l for l in lines if l.split("\t")[0].upper() not in {e.upper() for e in exclude}]
    (inst_dir / f"{name}.txt").write_text("\n".join(keep) + "\n")
    logger.info(f"instruments/{name}.txt: {len(keep)} symbols (excluded {sorted(exclude)})")
    return len(keep)


def write_future_calendar(qlib_dir: Path, extra_bdays: int = 60) -> Path:
    """Qlib's backtest needs calendars/day_future.txt (calendar extended past the last data day) to
    compute the end time of the final trading step. Extend with plain business days (holidays are irrelevant here)."""
    cal_dir = Path(qlib_dir) / "calendars"
    days = [l.strip() for l in (cal_dir / "day.txt").read_text().split() if l.strip()]
    last = pd.Timestamp(days[-1])
    future = pd.bdate_range(last + pd.Timedelta(days=1), periods=extra_bdays)
    out = days + [d.strftime("%Y-%m-%d") for d in future]
    path = cal_dir / "day_future.txt"
    path.write_text("\n".join(out) + "\n")
    logger.info(f"future calendar: {len(days)} real + {extra_bdays} future days -> {path}")
    return path
