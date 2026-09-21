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


def _rate_limited(symbols: list[str]) -> list[str]:
    """Symbols whose last yf.download error was a rate limit (yfinance stores per-ticker errors in shared._ERRORS)."""
    try:
        from yfinance import shared  # not re-exported at package level in yfinance 1.x
        errs = getattr(shared, "_ERRORS", {}) or {}
    except Exception:  # noqa: BLE001
        return list(symbols)  # cannot tell -> assume rate limit and back off
    return [s for s in symbols if s in errs and any(k in str(errs[s]) for k in ("Rate limit", "Too Many Requests", "429"))]


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
    stooq_fallback: bool = True,
) -> dict[str, Path]:
    """Download raw daily bars per symbol into out_dir/<SYM>.csv. Sequential requests (Yahoo rate-limits
    concurrent crumb fetches); on rate limit sleep backoff*attempt and retry the missing symbols only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = list(dict.fromkeys(symbols))
    todo = [s for s in symbols if not (skip_existing and (out_dir / f"{s}.csv").exists())]
    done = {s: out_dir / f"{s}.csv" for s in symbols if (out_dir / f"{s}.csv").exists()}
    logger.info(f"download: {len(symbols)} symbols, {len(todo)} to fetch, {len(done)} cached")
    n_chunks = (len(todo) - 1) // chunk_size + 1 if todo else 0
    for ci, i in enumerate(range(0, len(todo), chunk_size), start=1):
        remaining = todo[i : i + chunk_size]
        for attempt in range(1, retries + 1):
            df = None
            try:
                df = yf.download(
                    remaining, start=start, end=end, interval="1d", auto_adjust=False,
                    group_by="ticker", threads=False, progress=False, timeout=60,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"chunk {ci}/{n_chunks} attempt {attempt}: {e}")
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
            rl = _rate_limited(remaining)
            if rl or df is None:
                wait = backoff * attempt
                logger.warning(f"chunk {ci}/{n_chunks}: rate-limited/failed for {len(remaining)} symbols; sleeping {wait:.0f}s (attempt {attempt}/{retries})")
                time.sleep(wait)
            else:
                logger.warning(f"chunk {ci}/{n_chunks}: no Yahoo data for {remaining} (not a rate limit)")
                break
        if remaining and stooq_fallback:
            for sym in list(remaining):
                sub = download_stooq(sym, start, end)
                if sub is not None:
                    sub.to_csv(out_dir / f"{sym}.csv", index=False)
                    done[sym] = out_dir / f"{sym}.csv"
                    remaining.remove(sym)
                    logger.info(f"{sym}: filled from Stooq")
                time.sleep(1.0)
        logger.info(f"chunk {ci}/{n_chunks} done; total files {len(done)}")
        time.sleep(pause)
    failed = [s for s in symbols if s not in done]
    if failed:
        logger.warning(f"{len(failed)} symbols without data: {failed}")
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
