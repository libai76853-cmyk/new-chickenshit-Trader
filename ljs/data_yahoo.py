"""Download daily OHLCV from Yahoo Finance and normalize it into Qlib's CSV layout.

The normalization replicates qlib/scripts/data_collector/yahoo/collector.py `YahooNormalize1d`
(normalize_yahoo -> adjusted_price -> _manual_adj_data) so that the resulting Qlib data behaves like
data produced by Qlib's own collector:
  * rows with volume <= 0 are blanked (treated as non-trading days)
  * 100x price glitches (Yahoo cents/dollars mixups) are repaired
  * change = close / prev_close - 1 (on raw close)
  * factor = adjclose / close (forward filled); OHLC *= factor, volume /= factor
  * finally every field except change is scaled by the first valid close (Qlib convention), so
    raw_price == $close / $factor still holds.
Output columns: symbol,date,open,high,low,close,volume,factor,change
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

RAW_COLS = ["open", "high", "low", "close", "adjclose", "volume"]
OUT_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "factor", "change"]
_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adjclose", "Volume": "volume"}


def _frame_for(df: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    """Extract one ticker's frame from a yf.download(group_by='ticker') result."""
    if isinstance(df.columns, pd.MultiIndex):
        if sym not in df.columns.get_level_values(0):
            return None
        sub = df[sym]
    else:
        sub = df
    sub = sub.rename(columns=_RENAME)
    missing = [c for c in RAW_COLS if c not in sub.columns]
    if missing:
        logger.warning(f"{sym}: missing columns {missing}")
        return None
    sub = sub[RAW_COLS].dropna(how="all")
    if sub.empty:
        return None
    sub.index = pd.to_datetime(sub.index).tz_localize(None)
    sub.index.name = "date"
    return sub.reset_index()


class _YFCapture(logging.Handler):
    """Collect yfinance's per-ticker failure messages for one download() call.
    Formats seen (yfinance 1.7): "$CERN: possibly delisted; no timezone found",
    "$FB: Data doesn't exist for startDate = ...", "['A', 'B']: YFRateLimitError('Too Many Requests...')"."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.msgs: dict[str, str] = {}

    def emit(self, record: logging.LogRecord) -> None:
        m = record.getMessage().strip()
        hit = re.match(r"^\$?([A-Za-z0-9.\-^=]+): (.*)$", m)
        if hit:
            self.msgs[hit.group(1).upper()] = hit.group(2)
            return
        hit = re.match(r"^\[(.*?)\]: (.*)$", m, flags=re.S)
        if hit:
            for t in re.findall(r"'([^']+)'", hit.group(1)):
                self.msgs.setdefault(t.upper(), hit.group(2))


def _classify(msg: str | None) -> str:
    if not msg:
        return "unknown"
    if re.search(r"rate limit|too many requests|429", msg, flags=re.I):
        return "rate_limited"
    if re.search(r"delisted|no timezone|data doesn't exist|no data found|quote not found|not found for symbol", msg, flags=re.I):
        return "no_data"
    return "unknown"


def _download_chunk(symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame | None, dict[str, str]]:
    cap = _YFCapture()
    lg = logging.getLogger("yfinance")
    lg.addHandler(cap)
    try:
        df = yf.download(symbols, start=start, end=end, interval="1d", auto_adjust=False,
                         group_by="ticker", threads=False, progress=False, timeout=60)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"yf.download raised: {e}")
        df = None
        cap.msgs.setdefault("__ALL__", str(e))
    finally:
        lg.removeHandler(cap)
    return df, cap.msgs


def download_stooq(sym: str, start: str, end: str) -> pd.DataFrame | None:
    """Fallback source: Stooq daily CSV (split-adjusted, no separate adjusted close -> adjclose=close)."""
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d"
    try:
        import requests
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text.startswith("Date"):
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), parse_dates=["Date"])
    except Exception:  # noqa: BLE001
        return None
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["adjclose"] = df["close"]
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(end))]
    return df[["date"] + RAW_COLS] if not df.empty else None


def download_raw(
    symbols: Iterable[str],
    start: str,
    end: str,
    out_dir: Path,
    chunk_size: int = 20,
    retries: int = 5,
    pause: float = 2.0,
    backoff: float = 60.0,
    skip_existing: bool = True,
    stooq_fallback: bool = False,
    no_data_path: Path | None = None,
) -> dict[str, Path]:
    """Download raw daily bars per symbol into out_dir/<SYM>.csv.
    Sequential requests (Yahoo rate-limits concurrent crumb fetches). Failures are classified from yfinance's
    log messages: rate-limited -> sleep backoff*attempt and retry; no data (delisted/unknown symbol) -> give up
    at once and remember it in `no_data_path` so later runs skip it; unknown -> retry a couple of times.
    A chunk that returns nothing at all is treated as rate-limited."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = list(dict.fromkeys(symbols))
    known_no_data: set[str] = set()
    if no_data_path and Path(no_data_path).exists():
        known_no_data = {l.split("\t")[0] for l in Path(no_data_path).read_text().splitlines() if l.strip()}
    todo = [s for s in symbols if not (skip_existing and ((out_dir / f"{s}.csv").exists() or s in known_no_data))]
    done = {s: out_dir / f"{s}.csv" for s in symbols if (out_dir / f"{s}.csv").exists()}
    logger.info(f"download: {len(symbols)} symbols, {len(todo)} to fetch, {len(done)} cached, {len(known_no_data & set(symbols))} known no-data")
    n_chunks = (len(todo) - 1) // chunk_size + 1 if todo else 0
    no_data: dict[str, str] = {}
    if todo:  # warm-up: the first request of a process is often answered 429 on the cookie/crumb fetch
        try:
            yf.download("SPY", period="5d", interval="1d", progress=False, threads=False, timeout=30)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(pause)
    for ci, i in enumerate(range(0, len(todo), chunk_size), start=1):
        remaining = todo[i : i + chunk_size]
        unknown_tries: dict[str, int] = {}
        for attempt in range(1, retries + 1):
            df, msgs = _download_chunk(remaining, start, end)
            got: list[str] = []
            if df is not None and not df.empty:
                for sym in remaining:
                    sub = _frame_for(df, sym)
                    if sub is not None and not sub.empty:
                        sub.to_csv(out_dir / f"{sym}.csv", index=False)
                        done[sym] = out_dir / f"{sym}.csv"
                        got.append(sym)
            remaining = [s for s in remaining if s not in got]
            if not remaining:
                break
            kinds = {s: _classify(msgs.get(s) or msgs.get("__ALL__")) for s in remaining}
            if not got and all(k != "no_data" for k in kinds.values()):
                kinds = {s: "rate_limited" for s in remaining}  # whole chunk failed without a clear reason
            for s, k in kinds.items():
                if k == "no_data":
                    no_data[s] = msgs.get(s, "no data")
                elif k == "unknown":
                    unknown_tries[s] = unknown_tries.get(s, 0) + 1
                    if unknown_tries[s] >= 2:
                        no_data[s] = msgs.get(s, "unknown error, gave up")
            remaining = [s for s in remaining if s not in no_data]
            if not remaining:
                break
            if attempt < retries:
                wait = backoff * attempt
                logger.warning(f"chunk {ci}/{n_chunks}: {len(remaining)} symbols pending ({sorted(set(kinds[s] for s in remaining))}); retry in {wait:.0f}s (attempt {attempt}/{retries})")
                time.sleep(wait)
            else:
                logger.warning(f"chunk {ci}/{n_chunks}: giving up on {remaining}")
                for s in remaining:
                    no_data[s] = msgs.get(s, "retries exhausted")
        if remaining and stooq_fallback:
            for sym in list(remaining):
                sub = download_stooq(sym, start, end)
                if sub is not None:
                    sub.to_csv(out_dir / f"{sym}.csv", index=False)
                    done[sym] = out_dir / f"{sym}.csv"
                    remaining.remove(sym)
                    no_data.pop(sym, None)
                    logger.info(f"{sym}: filled from Stooq")
                time.sleep(1.0)
        logger.info(f"chunk {ci}/{n_chunks} done; files {len(done)}, no-data so far {len(no_data)}")
        time.sleep(pause)
    if no_data_path and no_data:
        p = Path(no_data_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d")
        with p.open("a") as fh:
            for s, why in sorted(no_data.items()):
                fh.write(f"{s}\t{stamp}\t{why[:80]}\n")
        logger.warning(f"{len(no_data)} symbols without data recorded in {p.name}: {sorted(no_data)[:15]}{' ...' if len(no_data) > 15 else ''}")
    return done


def load_calendar(bench_raw_csv: Path) -> pd.DatetimeIndex:
    b = pd.read_csv(bench_raw_csv, parse_dates=["date"])
    b = b[(b["volume"] > 0) & b["close"].notna()]
    return pd.DatetimeIndex(b["date"]).drop_duplicates().sort_values()


def normalize_1d(raw: pd.DataFrame, symbol: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Qlib YahooNormalize1d equivalent (see module docstring)."""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    cal = calendar[(calendar >= df.index.min()) & (calendar <= df.index.max())]
    df = df.reindex(cal)
    price_cols = ["open", "high", "low", "close", "adjclose"]
    bad = (df["volume"] <= 0) | df["volume"].isna()
    df.loc[bad, price_cols + ["volume"]] = np.nan
    for _ in range(10):  # repair 100x glitches, as Qlib does
        c = df["close"].ffill()
        ch = c / c.shift(1) - 1
        mask = (ch >= 89) & (ch <= 111)
        if not mask.any():
            break
        df.loc[mask, price_cols] = df.loc[mask, price_cols] / 100
    c = df["close"].ffill()
    df["change"] = c / c.shift(1) - 1
    df.loc[bad, ["open", "high", "low", "close", "volume", "change"]] = np.nan
    df["factor"] = (df["adjclose"] / df["close"]).ffill()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * df["factor"]
    df["volume"] = df["volume"] / df["factor"]
    fvi = df["close"].first_valid_index()
    if fvi is None:
        return pd.DataFrame(columns=OUT_COLS)
    first_close = float(df["close"].loc[fvi])
    for col in ["open", "high", "low", "close", "factor"]:
        df[col] = df[col] / first_close
    df["volume"] = df["volume"] * first_close
    df["symbol"] = symbol
    df.index.name = "date"
    return df.reset_index()[OUT_COLS]


def normalize_all(raw_dir: Path, out_dir: Path, calendar: pd.DatetimeIndex, min_rows: int = 250) -> list[str]:
    """Normalize every raw CSV; skip symbols with fewer than min_rows valid rows. Returns kept symbols."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    for p in sorted(raw_dir.glob("*.csv")):
        sym = p.stem
        raw = pd.read_csv(p)
        norm = normalize_1d(raw, sym, calendar)
        n_valid = int(norm["close"].notna().sum()) if not norm.empty else 0
        if n_valid < min_rows:
            logger.warning(f"{sym}: only {n_valid} valid rows, skipped")
            continue
        norm.to_csv(out_dir / f"{sym}.csv", index=False)
        kept.append(sym)
    logger.info(f"normalized {len(kept)} symbols -> {out_dir}")
    return kept
