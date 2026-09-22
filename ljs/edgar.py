"""Stage 3, step 1: structured 8-K events from SEC EDGAR (no NLP yet).

EDGAR's submissions API returns, per company (CIK), every filing with form type, filing date, acceptance
timestamp (to the second) and, for 8-Ks, the item codes (e.g. "2.02,9.01"). That is a typed event stream
for free. We build a (symbol, trading day) event panel from it and run event studies / signal tests on the
point-in-time universe. Laya (step 2) must add information beyond these item codes to be worth anything.

Point-in-time rule: a filing accepted at time tau becomes usable at the first close >= tau. Accepted before
16:00 ET on trading day D -> signal date D (trade at D+1 close, as in the Qlib label); accepted after 16:00
or on a non-trading day -> the next trading day.

SEC fair-access rules: declare a User-Agent with a contact and stay below 10 requests/second.
Set EDGAR_UA to override the default contact string.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
UA = os.environ.get("EDGAR_UA", "new-chickenshit-Trader research libai76853-cmyk@users.noreply.github.com")
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
MIN_INTERVAL = 0.2  # seconds between requests (5 req/s, half of SEC's limit)
_last_call = [0.0]

ITEM_GROUPS = {
    "ev_101": ["1.01"],            # entry into a material definitive agreement
    "ev_102": ["1.02"],            # termination of a material agreement
    "ev_103": ["1.03"],            # bankruptcy or receivership
    "ev_201": ["2.01"],            # completion of acquisition or disposition
    "ev_202": ["2.02"],            # results of operations (earnings)
    "ev_203": ["2.03"],            # creation of a direct financial obligation
    "ev_204": ["2.04"],            # triggering events (acceleration)
    "ev_205": ["2.05"],            # costs associated with exit or disposal
    "ev_206": ["2.06"],            # material impairments
    "ev_301": ["3.01"],            # delisting notice / failure to satisfy listing rule
    "ev_302": ["3.02"],            # unregistered sales of equity
    "ev_303": ["3.03"],            # material modification to rights of holders
    "ev_401": ["4.01"],            # change in auditor
    "ev_402": ["4.02"],            # non-reliance on previously issued financials (restatement)
    "ev_501": ["5.01"],            # change in control
    "ev_502": ["5.02"],            # departure/election of directors or officers
    "ev_503": ["5.03"],            # amendments to articles/bylaws; fiscal year change
    "ev_507": ["5.07"],            # submission of matters to a vote (annual meeting)
    "ev_701": ["7.01"],            # Regulation FD disclosure
    "ev_801": ["8.01"],            # other events
}
NEGATIVE_ITEMS = {"1.03", "2.04", "2.05", "2.06", "3.01", "4.01", "4.02"}


def _get(url: str, timeout: int = 30) -> requests.Response:
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    _last_call[0] = time.time()
    if r.status_code == 403 and "Rate Threshold" in r.text:
        logger.warning("SEC rate threshold exceeded; sleeping 600s")
        time.sleep(600)
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        _last_call[0] = time.time()
    return r


# ----------------------------------------------------------------------------- CIK mapping
def load_company_tickers(cache: Path, refresh: bool = False) -> dict[str, int]:
    """Yahoo-style ticker -> CIK from SEC's company_tickers.json (currently listed companies only)."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        return {k: int(v) for k, v in json.loads(cache.read_text()).items()}
    r = _get("https://www.sec.gov/files/company_tickers.json")
    r.raise_for_status()
    out = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in r.json().values()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def map_symbols_to_cik(symbols: list[str], universe: pd.DataFrame, tickers_json: dict[str, int]) -> tuple[dict[str, int], list[str]]:
    """Prefer Wikipedia CIK for current members (survives ticker changes), else SEC company_tickers."""
    wiki = {}
    if "cik" in universe.columns:
        for _, r in universe.iterrows():
            try:
                wiki[str(r["yahoo"]).upper()] = int(r["cik"])
            except (TypeError, ValueError):
                pass
    out, missing = {}, []
    for s in symbols:
        cik = wiki.get(s) or tickers_json.get(s)
        if cik:
            out[s] = int(cik)
        else:
            missing.append(s)
    return out, missing


# ----------------------------------------------------------------------------- submissions
def fetch_submissions(cik: int, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """All filings for a CIK (recent + older pages) as a DataFrame, cached as JSON per CIK."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"CIK{cik:010d}.parquet"
    if cpath.exists() and not refresh:
        return pd.read_parquet(cpath)
    r = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if r.status_code != 200:
        logger.warning(f"CIK {cik}: HTTP {r.status_code}")
        return pd.DataFrame()
    j = r.json()
    cols = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "items", "primaryDocument", "primaryDocDescription", "size"]
    frames = []
    rec = j["filings"]["recent"]
    frames.append(pd.DataFrame({c: rec.get(c, [None] * len(rec["form"])) for c in cols}))
    for f in j["filings"].get("files", []):
        rr = _get(f"https://data.sec.gov/submissions/{f['name']}")
        if rr.status_code == 200:
            old = rr.json()
            frames.append(pd.DataFrame({c: old.get(c, [None] * len(old["form"])) for c in cols}))
    df = pd.concat(frames, ignore_index=True)
    df["cik"] = cik
    df["name"] = j.get("name")
    df["tickers"] = ",".join(j.get("tickers") or [])
    df.to_parquet(cpath, index=False)
    return df


def fetch_all(cik_map: dict[str, int], cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    frames = []
    ciks = sorted(set(cik_map.values()))
    t0 = time.time()
    for i, cik in enumerate(ciks, 1):
        df = fetch_submissions(cik, cache_dir, refresh)
        if len(df):
            frames.append(df)
        if i % 50 == 0:
            logger.info(f"submissions {i}/{len(ciks)} ({(time.time()-t0)/60:.1f} min)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------------------------------------------------------- events
def signal_dates(accepted: pd.Series, calendar: pd.DatetimeIndex, tz: str = "US/Eastern", cutoff: str = "16:00") -> pd.Series:
    """First trading day whose close (cutoff, ET) is at/after the acceptance time.
    EDGAR's acceptanceDateTime strings end with 'Z' but the SEC documents them as Eastern time; `tz` says how
    to read the naive timestamp."""
    ts = pd.to_datetime(accepted.astype(str).str.replace("Z", "", regex=False), errors="coerce")
    ts = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward") if tz != "UTC" else ts.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    day = ts.dt.normalize().dt.tz_localize(None)
    after_close = ts.dt.strftime("%H:%M") >= cutoff
    cal = calendar.sort_values()
    pos = cal.searchsorted(day.values, side="left")  # first trading day >= day
    pos = np.where(after_close & (pos < len(cal)) & (cal.values[np.minimum(pos, len(cal) - 1)] == day.values), pos + 1, pos)
    pos = np.minimum(pos, len(cal) - 1)
    out = pd.Series(cal.values[pos], index=accepted.index)
    out[ts.isna()] = pd.NaT
    return out


def build_events(filings: pd.DataFrame, cik_map: dict[str, int], calendar: pd.DatetimeIndex, start: str, tz: str = "US/Eastern") -> pd.DataFrame:
    """One row per (symbol, filing): form, items list, acceptance time, signal date, item-group flags."""
    f = filings[filings["form"].isin(["8-K", "8-K/A"])].copy()
    f = f[pd.to_datetime(f["filingDate"]) >= pd.Timestamp(start) - pd.Timedelta(days=10)]
    cik2syms: dict[int, list[str]] = {}
    for s, c in cik_map.items():
        cik2syms.setdefault(c, []).append(s)
    f["symbols"] = f["cik"].map(cik2syms)
    f = f.explode("symbols").rename(columns={"symbols": "symbol"}).dropna(subset=["symbol"])
    f["signal_date"] = signal_dates(f["acceptanceDateTime"], calendar, tz=tz)
    f["item_list"] = f["items"].fillna("").astype(str).apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    for g, codes in ITEM_GROUPS.items():
        f[g] = f["item_list"].apply(lambda L, codes=codes: int(any(c in L for c in codes)))
    f["ev_neg"] = f["item_list"].apply(lambda L: int(any(c in NEGATIVE_ITEMS for c in L)))
    f["ev_any"] = 1
    f["is_amend"] = (f["form"] == "8-K/A").astype(int)
    f["n_items"] = f["item_list"].apply(len)
    return f.dropna(subset=["signal_date"]).reset_index(drop=True)


def daily_event_panel(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (symbol, signal_date): max of flags, count of filings."""
    flags = [c for c in events.columns if c.startswith("ev_")] + ["is_amend"]
    agg = {c: "max" for c in flags}
    agg["accessionNumber"] = "count"
    out = events.groupby(["symbol", "signal_date"]).agg(agg).rename(columns={"accessionNumber": "n_filings"}).reset_index()
    return out.rename(columns={"signal_date": "date"})
