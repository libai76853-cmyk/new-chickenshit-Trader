"""Stage 3 step 1: build the 8-K event panel from EDGAR and run event studies on the PIT universe.

Usage: .venv/bin/python scripts/04_edgar_events.py [--start 2010-01-01] [--eval-start 2022-01-01] [--tz US/Eastern] [--refresh]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ljs.edgar import load_company_tickers, map_symbols_to_cik, fetch_all, build_events, daily_event_panel, ITEM_GROUPS  # noqa: E402
from ljs.edgar_eval import daily_returns_panel, membership_mask, forward_returns, ew_series, event_study, pead_test, cross_sectional_signals, write_report  # noqa: E402
from ljs.kronos_features import load_panel, load_spells  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--eval-start", default="2022-01-01")
    ap.add_argument("--eval-end", default=None)
    ap.add_argument("--tz", default="UTC", help="EDGAR acceptanceDateTime is UTC (verified); US/Eastern only for what-if")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    edgar_dir = REPO / "data" / "edgar"
    edgar_dir.mkdir(parents=True, exist_ok=True)

    spells = load_spells(REPO / "data" / "membership_spells.csv")
    symbols = sorted(set(spells["symbol"]))
    uni = pd.DataFrame(json.loads((REPO / "data" / "universe_sp500.json").read_text()))
    tick = load_company_tickers(edgar_dir / "company_tickers.json", refresh=args.refresh)
    cik_map, missing = map_symbols_to_cik(symbols, uni, tick)
    logger.info(f"CIK mapped {len(cik_map)}/{len(symbols)} PIT symbols; missing {len(missing)}: {missing[:30]}")
    (edgar_dir / "cik_map.json").write_text(json.dumps({"map": cik_map, "missing": missing}, indent=1))

    filings = fetch_all(cik_map, edgar_dir / "submissions", refresh=args.refresh)
    logger.info(f"filings rows {len(filings):,}; forms top: {filings['form'].value_counts().head(6).to_dict()}")
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(REPO / "data" / "normalized" / "SPY.csv", usecols=["date"])["date"]))
    events = build_events(filings, cik_map, calendar, start=args.start, tz=args.tz)
    events.to_parquet(edgar_dir / "events_8k.parquet", index=False)
    panel_ev = daily_event_panel(events)
    logger.info(f"8-K events {len(events):,} ({events['symbol'].nunique()} symbols, {events['signal_date'].min().date()} -> {events['signal_date'].max().date()}); item flags: " +
                ", ".join(f"{g}={int(events[g].sum())}" for g in ITEM_GROUPS if events[g].sum() > 0))

    # returns / membership
    panel = load_panel(REPO / "data" / "normalized", symbols)
    rets_closes = pd.DataFrame({s: df["close"] for s, df in panel.items()}).reindex(calendar)
    mask = membership_mask(spells, calendar, list(rets_closes.columns))
    fwd = forward_returns(rets_closes, H=5)
    ew = {k: ew_series(v, mask) for k, v in fwd.items()}
    eval_end = args.eval_end or calendar[-1].strftime("%Y-%m-%d")
    flags = [g for g in ITEM_GROUPS] + ["ev_neg", "ev_any", "is_amend"]
    es, ev = event_study(panel_ev, fwd, ew, mask, flags, args.eval_start, eval_end)
    pead = pead_test(ev, "ev_202")
    cs = cross_sectional_signals(panel_ev, fwd, ew, mask, calendar, args.eval_start, eval_end, every=5)
    stamp = args.tag or dt.date.today().strftime("%Y%m%d")
    meta = {"n_filings": int(len(events)), "n_symbols": int(events["symbol"].nunique()), "start": str(events["signal_date"].min().date()), "end": str(events["signal_date"].max().date()),
            "eval_start": args.eval_start, "eval_end": eval_end, "tz": args.tz, "cik_mapped": len(cik_map), "cik_missing": missing}
    write_report(REPO / "reports" / f"edgar_8k_events_{stamp}.md", REPO / "reports" / f"edgar_8k_events_{stamp}.json",
                 f"8-K 结构化事件研究（PIT 标普 500，{args.eval_start} → {eval_end}）", es, pead, cs, meta)
    print(es.round(4).to_string())
    print("PEAD:", pead)
    print(cs.round(4).to_string())
    print(f"REPORT reports/edgar_8k_events_{stamp}.md")


if __name__ == "__main__":
    main()
