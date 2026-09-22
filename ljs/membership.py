"""Point-in-time S&P 500 membership reconstructed from Wikipedia.

Sources
  * current constituents with their `Date added`            -> ljs.universe (List of S&P 500 companies)
  * "Historical components of the S&P 500": one row per change (effective date, added ticker, removed ticker, reason)

Each ticker's add/remove events are walked forward in time to produce membership *spells* [start, end].
Rules
  * a removal with no earlier addition  -> member since the start of our window
  * an addition never removed and not a current member -> a rename or a missing row; dropped (the company is
    covered under its current ticker through `Date added`), unless the addition is within the last few days
  * the current list may lag the change table by a few days; a removal within the last few days wins
Ticker reuse: a delisted ticker can be recycled by a new company (e.g. FB in 2025). A spell is kept only if the
symbol's data starts well before the spell ends; spells are clamped to the available data range.
Output: DataFrame[symbol, start, end, source] and a Qlib instruments file with one line per spell.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

HIST_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
UA = {"User-Agent": "Mozilla/5.0 (ljs research script; contact: repo owner)"}
RECENT_DAYS = 10  # tolerance for the two Wikipedia pages being out of sync


def norm_ticker(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = re.sub(r"\[.*?\]", "", str(x)).strip().upper().replace(".", "-")
    s = re.sub(r"[^A-Z0-9\-]", "", s)  # stray pipes/footnote remnants such as "ALLE |"
    return s or None


def _parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip(), errors="coerce")


def fetch_changes_table() -> pd.DataFrame:
    r = requests.get(HIST_URL, headers=UA, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    t = next(t for t in tables if isinstance(t.columns, pd.MultiIndex) and "Effective Date" in t.columns.get_level_values(0))
    t = t.iloc[:, :6].copy()
    t.columns = ["date", "added", "added_name", "removed", "removed_name", "reason"]
    t["date"] = _parse_date(t["date"])
    t["added"] = t["added"].map(norm_ticker)
    t["removed"] = t["removed"].map(norm_ticker)
    for c in ("added_name", "removed_name", "reason"):
        t[c] = t[c].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip().replace({"nan": None})
    bad = t["date"].isna()
    if bad.any():
        logger.warning(f"changes table: {int(bad.sum())} rows with unparsable dates dropped")
    t = t[~bad].sort_values("date").reset_index(drop=True)
    return t


def load_changes(cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        df = pd.DataFrame(json.loads(cache_path.read_text()))
        df["date"] = pd.to_datetime(df["date"])
        return df
    df = fetch_changes_table()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    cache_path.write_text(json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=1))
    return df


def classify_reason(reason: str | None) -> str:
    r = reason.lower() if isinstance(reason, str) else ""  # NaN-safe (pandas 3 string dtype)
    if re.search(r"acquir|merg|taken private|purchas|bought|combin|takeover", r):
        return "acquisition"
    if "market cap" in r:
        return "demotion"
    if re.search(r"bankrupt|chapter 11|delist|liquidat", r):
        return "bankruptcy/delisting"
    if re.search(r"spin|split|reorgan|separat", r):
        return "corporate action"
    return "other"


def build_spells(universe: pd.DataFrame, changes: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Membership spells within [start, end] for every ticker that was a member at some point."""
    START, END = pd.Timestamp(start), pd.Timestamp(end)
    one_day = pd.Timedelta(days=1)
    current = {norm_ticker(s) for s in universe["yahoo"]}
    date_added = {norm_ticker(r["yahoo"]): pd.to_datetime(str(r["date_added"])[:10], errors="coerce") for _, r in universe.iterrows()}

    events: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for _, r in changes.iterrows():
        a, rm = norm_ticker(r["added"]), norm_ticker(r["removed"])  # pandas 3 turns None into NaN
        if a:
            events.setdefault(a, []).append((pd.Timestamp(r["date"]), "add"))
        if rm:
            events.setdefault(rm, []).append((pd.Timestamp(r["date"]), "remove"))
    current.discard(None)

    rows, diag = [], {"renamed_or_missing_removal": [], "removed_without_addition": [], "readded_without_event": [], "current_removed_recently": []}
    for sym in sorted(current | set(events)):
        evs = sorted(events.get(sym, []))
        is_current = sym in current
        open_start = None
        spells: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
        for i, (d, kind) in enumerate(evs):
            if kind == "add":
                if open_start is None:
                    open_start = d
            else:  # remove
                if open_start is None:
                    if not spells:
                        diag["removed_without_addition"].append(sym)
                    spells.append((START, d - one_day, "since_window_start"))
                else:
                    spells.append((open_start, d - one_day, "events"))
                    open_start = None
        last_kind, last_date = (evs[-1][1], evs[-1][0]) if evs else (None, None)
        if open_start is not None:
            if is_current or (END - open_start).days <= RECENT_DAYS:
                spells.append((open_start, END, "events"))
            else:
                diag["renamed_or_missing_removal"].append(sym)
        elif is_current:
            if last_kind == "remove" and (END - last_date).days <= RECENT_DAYS:
                diag["current_removed_recently"].append(sym)  # change table is ahead of the constituents table
            else:
                da = date_added.get(sym)
                s0 = da if (da is not None and not pd.isna(da)) else START
                if last_kind == "remove":
                    s0 = max(s0, last_date + one_day)
                    diag["readded_without_event"].append(sym)
                spells.append((s0, END, "date_added" if last_kind is None else "readded"))
        for s0, s1, src in spells:
            s0, s1 = max(s0, START), min(s1, END)
            if s0 <= s1:
                rows.append({"symbol": sym, "start": s0, "end": s1, "source": src})
    spells_df = pd.DataFrame(rows).sort_values(["symbol", "start"]).reset_index(drop=True)
    diag = {k: sorted(set(v)) for k, v in diag.items()}
    return spells_df, diag


def data_ranges(norm_dir: Path) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """First/last date with a valid close per normalized CSV."""
    out = {}
    for p in sorted(Path(norm_dir).glob("*.csv")):
        df = pd.read_csv(p, usecols=["date", "close"])
        df = df[df["close"].notna()]
        if len(df):
            out[p.stem.upper()] = (pd.Timestamp(df["date"].iloc[0]), pd.Timestamp(df["date"].iloc[-1]))
    return out


def filter_spells_by_data(spells: pd.DataFrame, ranges: dict, reuse_tolerance_days: int = 30) -> pd.DataFrame:
    """Keep spells backed by data; reject probable ticker reuse; clamp to the data range. Adds `kept`, `why` columns."""
    tol = pd.Timedelta(days=reuse_tolerance_days)
    out = spells.copy()
    kept, why, s_new, e_new = [], [], [], []
    for _, r in out.iterrows():
        rng = ranges.get(r["symbol"])
        if rng is None:
            kept.append(False); why.append("no_data"); s_new.append(r["start"]); e_new.append(r["end"]); continue
        first, last = rng
        if first > r["end"] - tol:
            kept.append(False); why.append("ticker_reuse_or_late_data"); s_new.append(r["start"]); e_new.append(r["end"]); continue
        if last < r["start"]:
            kept.append(False); why.append("data_ends_before_spell"); s_new.append(r["start"]); e_new.append(r["end"]); continue
        s0, s1 = max(r["start"], first), min(r["end"], last)
        kept.append(True); why.append("ok" if (s0 == r["start"] and s1 == r["end"]) else "clamped"); s_new.append(s0); e_new.append(s1)
    out["kept"], out["why"], out["start_eff"], out["end_eff"] = kept, why, s_new, e_new
    return out


def write_pit_instruments(qlib_dir: Path, spells: pd.DataFrame, name: str) -> int:
    inst_dir = Path(qlib_dir) / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    k = spells[spells["kept"]]
    lines = [f"{r['symbol']}\t{r['start_eff']:%Y-%m-%d}\t{r['end_eff']:%Y-%m-%d}" for _, r in k.iterrows()]
    (inst_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
    logger.info(f"instruments/{name}.txt: {len(lines)} spells over {k['symbol'].nunique()} symbols")
    return len(lines)


def coverage_summary(spells: pd.DataFrame, changes: pd.DataFrame, current: set[str], start: str, end: str) -> dict:
    """How much of the true membership history is backed by data, and what kind of removals are missing."""
    START, END = pd.Timestamp(start), pd.Timestamp(end)
    ch = changes[(changes["date"] >= START) & (changes["date"] <= END)].copy()
    ch["category"] = ch["reason"].map(classify_reason)
    ch["removed"] = ch["removed"].map(norm_ticker)
    ch["added"] = ch["added"].map(norm_ticker)
    removed = ch[ch["removed"].notna()]
    have = set(spells.loc[spells["kept"], "symbol"])
    removed_syms = set(removed["removed"])
    by_cat = removed.groupby("category")["removed"].agg(lambda s: (len(set(s)), len(set(s) & have)))
    member_days = lambda df: int(((df["end_eff"] - df["start_eff"]).dt.days + 1).sum()) if len(df) else 0
    all_days = int(((spells["end"] - spells["start"]).dt.days + 1).sum())
    return {
        "window": [start, end],
        "change_rows": int(len(ch)),
        "additions": int(ch["added"].notna().sum()),
        "removals": int(len(removed)),
        "distinct_removed_tickers": len(removed_syms),
        "removed_tickers_with_data": len(removed_syms & have),
        "removed_by_category": {k: {"tickers": int(v[0]), "with_data": int(v[1])} for k, v in by_cat.items()},
        "spells_total": int(len(spells)),
        "spells_kept": int(spells["kept"].sum()),
        "spells_dropped_by_reason": spells.loc[~spells["kept"], "why"].value_counts().to_dict(),
        "member_days_total": all_days,
        "member_days_with_data": member_days(spells[spells["kept"]]),
        "symbols_total": int(spells["symbol"].nunique()),
        "symbols_kept": int(spells.loc[spells["kept"], "symbol"].nunique()),
        "current_members": len(current),
    }
