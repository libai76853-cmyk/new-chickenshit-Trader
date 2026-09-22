"""Stage 3 step 2: Laya zero-shot on 8-K text.

score:    .venv/bin/python scripts/05_laya_8k.py score --start 2024-07-01 --end 2026-09-21 --groups ev_202,ev_502,ev_801,ev_101,ev_701,ev_neg --out data/edgar/laya_scores.parquet [--limit N]
evaluate: .venv/bin/python scripts/05_laya_8k.py evaluate --scores data/edgar/laya_scores.parquet --tag post2024h2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--start", required=True); s.add_argument("--end", required=True)
    s.add_argument("--groups", default="ev_202,ev_502,ev_801,ev_101,ev_701,ev_neg")
    s.add_argument("--out", default="data/edgar/laya_scores.parquet"); s.add_argument("--limit", type=int, default=None)
    s.add_argument("--device", default=None)
    e = sub.add_parser("evaluate")
    e.add_argument("--scores", default="data/edgar/laya_scores.parquet"); e.add_argument("--tag", required=True)
    e.add_argument("--eval-start", default="2022-01-01")
    args = ap.parse_args()
    from ljs.laya_events import select_events, score_events, evaluate
    from ljs.kronos_features import load_spells, load_panel

    edgar_dir = REPO / "data" / "edgar"
    events = pd.read_parquet(edgar_dir / "events_8k.parquet")
    events["signal_date"] = pd.to_datetime(events["signal_date"])
    spells = load_spells(REPO / "data" / "membership_spells.csv")
    if args.cmd == "score":
        ev = select_events(events, args.start, args.end, args.groups.split(","), spells)
        if args.limit:
            ev = ev.sample(n=min(args.limit, len(ev)), random_state=0).sort_values("signal_date")
        logger.info(f"selected {len(ev)} filings; groups: " + ", ".join(f"{g}={int(ev[g].sum())}" for g in args.groups.split(",")))
        score_events(ev, REPO / args.out, edgar_dir / "docs", device=args.device)
    else:
        from ljs.edgar import daily_event_panel
        from ljs.edgar_eval import membership_mask, forward_returns, ew_series, event_study
        scored = pd.read_parquet(REPO / args.scores)
        symbols = sorted(set(spells["symbol"]))
        panel = load_panel(REPO / "data" / "normalized", symbols)
        calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(REPO / "data" / "normalized" / "SPY.csv", usecols=["date"])["date"]))
        closes = pd.DataFrame({s_: df["close"] for s_, df in panel.items()}).reindex(calendar)
        mask = membership_mask(spells, calendar, list(closes.columns))
        fwd = forward_returns(closes, H=5)
        ew = {k: ew_series(v, mask) for k, v in fwd.items()}
        # per-filing abnormal returns: reuse event_study's per-event frame on a filing-level panel
        per_filing = events.rename(columns={"signal_date": "date"})
        flags = [c for c in per_filing.columns if c.startswith("ev_")]
        _, evr = event_study(per_filing, fwd, ew, mask, flags, args.eval_start, calendar[-1].strftime("%Y-%m-%d"))
        res = evaluate(scored, evr, REPO / "reports" / f"laya_8k_{args.tag}.md", REPO / "reports" / f"laya_8k_{args.tag}.json", f"Laya 零样本读 8-K 正文：事件内检验（{args.tag}）")
        o = res["overall"]
        print(f"REPORT reports/laya_8k_{args.tag}.md")
        print(f"laya_dir: vs reaction {o['laya_dir']['sp_ab_r0']:.4f} | vs drift {o['laya_dir']['sp_ab_r_exec']:.4f} | resid vs drift {o['laya_resid']['sp_ab_r_exec']:.4f} | tercile t {o['laya_dir_tercile']['t_monthly']:.2f}")


if __name__ == "__main__":
    main()
