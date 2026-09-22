"""Event studies and signal tests for the 8-K event panel on the point-in-time universe.

Conventions (same as the rest of the repo): signal known by close of day t; r_exec = close[t+6]/close[t+1]-1
(enter next close, exit five trading days later); r1 = close[t+2]/close[t+1]-1; r0 = close[t]/close[t-1]-1 is
the reaction on the signal day itself. Abnormal returns subtract the equal-weight return of that day's
point-in-time members (EW-PIT). t-statistics cluster by date: average within date, then t across dates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ljs.kronos_features import load_panel, load_spells, members_on  # reuse data helpers

PRICE = "close"


def daily_returns_panel(panel: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide DataFrame of daily simple returns (date x symbol) aligned to the trading calendar."""
    closes = pd.DataFrame({s: df[PRICE] for s, df in panel.items()}).reindex(calendar)
    return closes.pct_change(fill_method=None)


def membership_mask(spells: pd.DataFrame, calendar: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(False, index=calendar, columns=symbols)
    for _, r in spells.iterrows():
        if r["symbol"] in m.columns:
            m.loc[r["start_eff"] : r["end_eff"], r["symbol"]] = True
    return m


def forward_returns(closes: pd.DataFrame, H: int = 5) -> dict[str, pd.DataFrame]:
    """Wide frames keyed by signal date t."""
    c = closes
    return {
        "r0": c / c.shift(1) - 1,                       # reaction on t (close t-1 -> close t)
        "r1": c.shift(-2) / c.shift(-1) - 1,            # t+1 -> t+2
        "r_exec": c.shift(-(H + 1)) / c.shift(-1) - 1,  # t+1 -> t+H+1
    }


def ew_series(wide: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    return wide.where(mask).mean(axis=1)


def event_study(events: pd.DataFrame, fwd: dict[str, pd.DataFrame], ew: dict[str, pd.Series], mask: pd.DataFrame,
                flags: list[str], start: str, end: str) -> pd.DataFrame:
    """Per item-group: number of events (member-days), mean abnormal r0 / r1 / r_exec, date-clustered t-stats."""
    ev = events[(events["date"] >= pd.Timestamp(start)) & (events["date"] <= pd.Timestamp(end))].copy()
    ev = ev[ev["symbol"].isin(mask.columns)]
    # membership at signal date
    ev["member"] = [bool(mask.at[d, s]) if d in mask.index else False for d, s in zip(ev["date"], ev["symbol"])]
    ev = ev[ev["member"]]
    for k in ("r0", "r1", "r_exec"):
        vals = np.array([fwd[k].at[d, s] if (d in fwd[k].index) else np.nan for d, s in zip(ev["date"], ev["symbol"])], dtype=float)
        ev[k] = vals
        ev[f"ab_{k}"] = vals - ew[k].reindex(ev["date"]).values
    rows = []
    for flag in flags:
        sub = ev[ev[flag] == 1].dropna(subset=["ab_r_exec"])
        if len(sub) < 20:
            continue
        rec = {"flag": flag, "n_events": int(len(sub)), "n_dates": int(sub["date"].nunique())}
        for k in ("r0", "r1", "r_exec"):
            x = sub.dropna(subset=[f"ab_{k}"])
            by_date = x.groupby("date")[f"ab_{k}"].mean()
            rec[f"ab_{k}_mean"] = float(x[f"ab_{k}"].mean())
            rec[f"ab_{k}_t"] = float(by_date.mean() / by_date.std() * np.sqrt(len(by_date))) if len(by_date) > 2 else np.nan
            rec[f"ab_{k}_pos"] = float((x[f"ab_{k}"] > 0).mean())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("flag"), ev


def pead_test(ev: pd.DataFrame, flag: str = "ev_202", q: int = 3) -> dict:
    """Post-announcement drift: among earnings 8-Ks, does the sign/size of the reaction (ab_r0) predict ab_r_exec?"""
    sub = ev[(ev[flag] == 1)].dropna(subset=["ab_r0", "ab_r_exec"]).copy()
    out = {"n_events": int(len(sub))}
    if len(sub) < 50:
        return out
    # cross-event rank correlation, pooled and date-clustered where dates have >= 5 events
    out["spearman_pooled"] = float(sub["ab_r0"].corr(sub["ab_r_exec"], method="spearman"))
    g = sub.groupby("date")
    per_date = g.apply(lambda x: x["ab_r0"].corr(x["ab_r_exec"], method="spearman") if len(x) >= 5 else np.nan).dropna()
    out["spearman_by_date_mean"] = float(per_date.mean()) if len(per_date) else np.nan
    out["spearman_by_date_t"] = float(per_date.mean() / per_date.std() * np.sqrt(len(per_date))) if len(per_date) > 2 else np.nan
    out["n_dates_ge5"] = int(len(per_date))
    # tercile spread pooled over all events (top reaction minus bottom reaction), t by event with date clustering
    sub["bucket"] = pd.qcut(sub["ab_r0"].rank(method="first"), q, labels=False)
    top, bot = sub[sub["bucket"] == q - 1], sub[sub["bucket"] == 0]
    out["top_minus_bottom_ab_r_exec"] = float(top["ab_r_exec"].mean() - bot["ab_r_exec"].mean())
    # cluster by month to get a rough t
    m_top = top.groupby(top["date"].dt.to_period("M"))["ab_r_exec"].mean()
    m_bot = bot.groupby(bot["date"].dt.to_period("M"))["ab_r_exec"].mean()
    d = (m_top - m_bot).dropna()
    out["top_minus_bottom_t_monthly"] = float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 2 else np.nan
    out["n_months"] = int(len(d))
    return out


def cross_sectional_signals(events: pd.DataFrame, fwd: dict[str, pd.DataFrame], ew: dict[str, pd.Series], mask: pd.DataFrame,
                            calendar: pd.DatetimeIndex, start: str, end: str, every: int = 5) -> pd.DataFrame:
    """Daily cross-sectional RankIC of event-derived features over all PIT members (zeros where no event).
    Features: count of 8-Ks in the last 5 / 20 trading days, negative-item count in last 20 days,
    earnings-8-K reaction in last 5 days (NaN if none), days since last 8-K (capped 60)."""
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))][::every]
    syms = list(mask.columns)
    ev_any = events.pivot_table(index="date", columns="symbol", values="ev_any", aggfunc="max").reindex(index=calendar, columns=syms).fillna(0.0)
    ev_neg = events.pivot_table(index="date", columns="symbol", values="ev_neg", aggfunc="max").reindex(index=calendar, columns=syms).fillna(0.0)
    ev_202 = events.pivot_table(index="date", columns="symbol", values="ev_202", aggfunc="max").reindex(index=calendar, columns=syms).fillna(0.0)
    react = fwd["r0"].sub(ew["r0"], axis=0).reindex(index=calendar, columns=syms).where(ev_202 == 1)
    feats = {
        "n8k_5d": ev_any.rolling(5, min_periods=1).sum(),
        "n8k_20d": ev_any.rolling(20, min_periods=1).sum(),
        "neg_20d": ev_neg.rolling(20, min_periods=1).sum(),
        "pead_5d": react.ffill(limit=4),  # most recent earnings reaction within the last 5 days
    }
    last_idx = pd.Series(np.arange(len(calendar)), index=calendar)
    had = ev_any.where(ev_any > 0)
    last_pos = had.mul(last_idx.values[:, None]).ffill()
    feats["days_since_8k"] = (last_idx.values[:, None] - last_pos).clip(upper=60).fillna(60)
    rows = []
    for name, F in feats.items():
        for tgt in ("r_exec", "r1"):
            ics = []
            for d in dates:
                if d not in F.index:
                    continue
                m = mask.loc[d]
                x = F.loc[d][m]
                y = fwd[tgt].loc[d][m]
                ok = x.notna() & y.notna()
                if ok.sum() < 30 or x[ok].nunique() < 3:
                    continue
                ics.append(x[ok].corr(y[ok], method="spearman"))
            ics = pd.Series(ics)
            if len(ics) > 5:
                rows.append({"feature": name, "target": tgt, "rank_ic": float(ics.mean()), "t": float(ics.mean() / ics.std() * np.sqrt(len(ics))), "n_dates": int(len(ics)), "pos": float((ics > 0).mean())})
    return pd.DataFrame(rows)


def write_report(path_md: Path, path_json: Path, title: str, es: pd.DataFrame, pead: dict, cs: pd.DataFrame, meta: dict) -> None:
    res = {"title": title, "meta": meta, "event_study": es.reset_index().to_dict(orient="records"), "pead": pead, "cross_sectional": cs.to_dict(orient="records")}
    path_json.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    L = [f"# {title}\n", f"事件面板：{meta['n_filings']:,} 份 8-K（{meta['n_symbols']} 只股票，{meta['start']} → {meta['end']}）；评估期 {meta['eval_start']} → {meta['eval_end']}，仅计入信号日为 PIT 成员的事件。异常收益 = 个股收益 − 当日 PIT 等权收益；t 值按日聚类。\n",
         "## 事件研究：按 8-K 项目\n", "| 项目 | 事件数 | 日数 | 信号日反应 ab_r0 | t | 次日 ab_r1 | t | 5 日 ab_r_exec | t | 5 日为正占比 |\n|---|---|---|---|---|---|---|---|---|---|"]
    for flag, r in es.iterrows():
        L.append(f"| {flag} | {int(r['n_events'])} | {int(r['n_dates'])} | {r['ab_r0_mean']*100:.2f}% | {r['ab_r0_t']:.1f} | {r['ab_r1_mean']*100:.2f}% | {r['ab_r1_t']:.1f} | {r['ab_r_exec_mean']*100:.2f}% | {r['ab_r_exec_t']:.1f} | {r['ab_r_exec_pos']*100:.0f}% |")
    L.append("\n## 业绩公告后漂移（PEAD）：2.02 事件中，信号日反应能否预测后 5 日收益\n")
    for k, v in pead.items():
        L.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    L.append("\n## 全横截面 RankIC（每 5 个交易日，全体 PIT 成员，无事件记 0）\n", )
    L.append("| 特征 | 目标 | RankIC | t | 日数 | 正占比 |\n|---|---|---|---|---|---|")
    for _, r in cs.iterrows():
        L.append(f"| {r['feature']} | {r['target']} | {r['rank_ic']:.4f} | {r['t']:.2f} | {r['n_dates']} | {r['pos']*100:.0f}% |")
    path_md.write_text("\n".join(L) + "\n", encoding="utf-8")
