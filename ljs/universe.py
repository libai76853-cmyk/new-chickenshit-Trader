"""Universe definition: S&P 500 constituents (current list from Wikipedia) plus a benchmark ETF.

CAVEAT (survivorship bias): the list is the *current* membership. Backtests over past years therefore
exclude companies that were removed (bankrupt, acquired, demoted) and include ones that were not yet members.
This biases returns upward. A point-in-time membership history is the proper fix (later stage).
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = {"User-Agent": "Mozilla/5.0 (ljs research script; contact: repo owner)"}
BENCHMARK = "SPY"  # also used as the trading calendar source


def to_yahoo_symbol(sym: str) -> str:
    """Wikipedia uses 'BRK.B'; Yahoo uses 'BRK-B'."""
    return sym.strip().upper().replace(".", "-")


def to_qlib_symbol(sym: str) -> str:
    """Qlib instrument codes: keep Yahoo form but uppercase (dump_bin uppercases file stems)."""
    return to_yahoo_symbol(sym)


def fetch_sp500_table(retries: int = 3, pause: float = 2.0) -> pd.DataFrame:
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            r = requests.get(WIKI_URL, headers=UA, timeout=30)
            r.raise_for_status()
            tables = pd.read_html(io.StringIO(r.text))
            t = tables[0]
            if "Symbol" not in t.columns:
                raise ValueError(f"unexpected table columns: {t.columns.tolist()}")
            return t
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(pause)
    raise RuntimeError(f"failed to fetch S&P 500 table: {last_err}")


def load_universe(cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    """Return DataFrame[symbol, yahoo, security, sector, sub_industry, date_added, cik]; cached as JSON."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.DataFrame(json.loads(cache_path.read_text()))
    t = fetch_sp500_table()
    df = pd.DataFrame(
        {
            "symbol": t["Symbol"].astype(str).str.strip().str.upper(),
            "yahoo": t["Symbol"].astype(str).map(to_yahoo_symbol),
            "security": t["Security"].astype(str),
            "sector": t["GICS Sector"].astype(str),
            "sub_industry": t["GICS Sub-Industry"].astype(str),
            "date_added": t["Date added"].astype(str),
            "cik": pd.to_numeric(t["CIK"], errors="coerce").astype("Int64") if "CIK" in t.columns else pd.array([None] * len(t), dtype="Int64"),
        }
    ).drop_duplicates("yahoo").sort_values("yahoo").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=1))
    return df
