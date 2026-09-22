"""Stage 3, step 2: Laya zero-shot reading of 8-K text -> typed probabilities -> event features -> tests.

For each selected 8-K (from data/edgar/events_8k.parquet) we fetch the informative text (press release for
2.02/7.01/8.01, Item sections otherwise), ask Laya a fixed set of typed questions, and store the probabilities.
Evaluation asks three things, all within the scored events and with EW-PIT abnormal returns:
  (a) does Laya read the news the way the market did?      Spearman(laya_dir, ab_r0)   [comprehension check]
  (b) does it predict the post-filing drift?                 Spearman(laya_dir, ab_r_exec)
  (c) is there anything beyond the reaction and item codes?  Spearman(resid(laya_dir | ab_r0, items), ab_r_exec)
Only (b)/(c) are tradable under the repo's close-to-close convention.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]

QUESTIONS = {
    "direction": {"type": "choice", "instructions": "Is the company news in `text` positive, negative or neutral for the company's stock price over the next week?",
                  "criteria": {"positive": "good news: results beat expectations, guidance raised, buybacks or dividends increased, favorable deals or approvals",
                               "negative": "bad news: results miss, guidance cut, impairments, restatements, investigations, unexpected executive departures, dilutive financing",
                               "neutral": "routine, procedural or immaterial disclosure"}},
    "guidance": {"type": "choice", "instructions": "What does `text` say about the company's financial guidance or outlook?",
                 "criteria": {"raised": "guidance or outlook raised", "lowered": "guidance or outlook lowered or withdrawn", "maintained": "guidance reaffirmed or unchanged", "none": "guidance not discussed"}},
    "results": {"type": "choice", "instructions": "How do the reported results in `text` compare with expectations?",
                "criteria": {"beat": "better than expected or strong growth", "miss": "worse than expected, declines or losses", "inline": "roughly as expected", "none": "no results reported"}},
    "exec_change": {"type": "noul", "instructions": "Does `text` announce that a senior executive (CEO, CFO, COO, president) is leaving or being replaced?"},
    "mna": {"type": "noul", "instructions": "Does `text` announce an acquisition, merger, divestiture or major strategic transaction?"},
    "restatement": {"type": "noul", "instructions": "Does `text` say previously issued financial statements should not be relied upon, or announce a restatement or accounting investigation?"},
    "importance": {"type": "score", "instructions": "How material is the news in `text` for investors?", "criteria": ["immaterial", "minor", "moderate", "major", "critical"]},
}
FEATURES = ["laya_dir", "p_pos", "p_neg", "p_neutral", "guid_net", "res_net", "p_exec", "p_mna", "p_restate", "importance"]


def load_agent(device: str | None = None):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import laya

    agent = laya.load("convaiinnovations/laya", device=device)
    logger.info(f"Laya loaded on {agent.device}")
    return agent


def features_from_answers(ans: dict) -> dict:
    d = ans["direction"]["probabilities"]
    g = ans["guidance"]["probabilities"]
    r = ans["results"]["probabilities"]
    return {
        "laya_dir": float(d.get("positive", 0) - d.get("negative", 0)),
        "p_pos": float(d.get("positive", 0)), "p_neg": float(d.get("negative", 0)), "p_neutral": float(d.get("neutral", 0)),
        "guid_net": float(g.get("raised", 0) - g.get("lowered", 0)),
        "res_net": float(r.get("beat", 0) - r.get("miss", 0)),
        "p_exec": float(ans["exec_change"]["noul"]), "p_mna": float(ans["mna"]["noul"]), "p_restate": float(ans["restatement"]["noul"]),
        "importance": float(ans["importance"]["score"]),
    }


def select_events(events: pd.DataFrame, start: str, end: str, groups: list[str], spells: pd.DataFrame) -> pd.DataFrame:
    """8-K filings in [start, end] whose symbol was a PIT member on the signal date, restricted to item groups."""
    ev = events[(events["signal_date"] >= pd.Timestamp(start)) & (events["signal_date"] <= pd.Timestamp(end))].copy()
    ev = ev[ev[groups].sum(axis=1) > 0]
    sp = spells.set_index("symbol")
    keep = []
    for _, r in ev.iterrows():
        s = sp.loc[[r["symbol"]]] if r["symbol"] in sp.index else None
        keep.append(bool(s is not None and ((s["start_eff"] <= r["signal_date"]) & (s["end_eff"] >= r["signal_date"])).any()))
    ev = ev[np.array(keep)]
    return ev.drop_duplicates("accessionNumber").reset_index(drop=True)


def score_events(ev: pd.DataFrame, out_path: Path, docs_dir: Path, device: str | None = None, checkpoint_every: int = 200, max_chars: int = 3000) -> Path:
    """Fetch text + Laya probabilities for every filing in `ev`; resumable via out_path parquet."""
    from ljs.edgar_text import event_text

    out_path = Path(out_path)
    done = set()
    frames = []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        frames.append(prev)
        done = set(prev["accessionNumber"])
        logger.info(f"resuming: {len(done)} filings already scored")
    todo = ev[~ev["accessionNumber"].isin(done)]
    logger.info(f"scoring {len(todo)} filings ({len(done)} cached)")
    agent = load_agent(device)
    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(todo.iterrows(), 1):
        et = event_text(int(r["cik"]), r["accessionNumber"], list(r["item_list"]), docs_dir, primary=r.get("primaryDocument"), max_chars=max_chars)
        rec = {"accessionNumber": r["accessionNumber"], "symbol": r["symbol"], "signal_date": r["signal_date"], "cik": int(r["cik"]),
               "items": ",".join(r["item_list"]), "text_source": et["source"], "text_chars": len(et["text"])}
        if len(et["text"]) >= 200:
            try:
                ans = agent.predict({"text": et["text"]}, QUESTIONS)["answers"]
                rec.update(features_from_answers(ans))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{r['accessionNumber']}: laya failed: {e}")
        rows.append(rec)
        if i % 50 == 0:
            logger.info(f"[{i}/{len(todo)}] {(time.time()-t0)/60:.1f} min")
        if i % checkpoint_every == 0:
            pd.concat(frames + [pd.DataFrame(rows)], ignore_index=True).to_parquet(out_path, index=False)
    if rows:
        frames.append(pd.DataFrame(rows))
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
    logger.info(f"saved {out_path} ({(time.time()-t0)/60:.1f} min)")
    return out_path


def _sp(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(a[ok].corr(b[ok], method="spearman")) if ok.sum() > 20 else np.nan


def _monthly_t(x: pd.Series, dates: pd.Series) -> float:
    m = x.groupby(dates.dt.to_period("M")).mean().dropna()
    return float(m.mean() / m.std() * np.sqrt(len(m))) if len(m) > 2 else np.nan


def evaluate(scored: pd.DataFrame, ev_returns: pd.DataFrame, out_md: Path, out_json: Path, title: str) -> dict:
    """scored: laya features per filing; ev_returns: per (accessionNumber) ab_r0/ab_r1/ab_r_exec from ljs.edgar_eval.event_study."""
    df = scored.merge(ev_returns[["accessionNumber", "symbol", "ab_r0", "ab_r1", "ab_r_exec"] + [c for c in ev_returns.columns if c.startswith("ev_")]].drop_duplicates("accessionNumber"),
                      on=["accessionNumber", "symbol"], how="inner")
    df = df.dropna(subset=["laya_dir", "ab_r_exec"])
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    res = {"title": title, "n_scored": int(len(scored)), "n_with_text": int(scored["laya_dir"].notna().sum()), "n_eval": int(len(df)),
           "date_range": [str(df["signal_date"].min().date()), str(df["signal_date"].max().date())], "by_group": {}, "overall": {}}
    feats = ["laya_dir", "guid_net", "res_net", "p_exec", "p_mna", "p_restate", "importance"]
    for f in feats:
        res["overall"][f] = {"sp_ab_r0": _sp(df[f], df["ab_r0"]), "sp_ab_r1": _sp(df[f], df["ab_r1"]), "sp_ab_r_exec": _sp(df[f], df["ab_r_exec"])}
    # tercile spread on laya_dir, pooled, monthly-cluster t
    q = pd.qcut(df["laya_dir"].rank(method="first"), 3, labels=False)
    top, bot = df[q == 2], df[q == 0]
    res["overall"]["laya_dir_tercile"] = {"top_minus_bottom_ab_r_exec": float(top["ab_r_exec"].mean() - bot["ab_r_exec"].mean()),
                                          "top_minus_bottom_ab_r0": float(top["ab_r0"].mean() - bot["ab_r0"].mean()),
                                          "t_monthly": _monthly_t(pd.concat([top["ab_r_exec"], -bot["ab_r_exec"]]), pd.concat([top["signal_date"], bot["signal_date"]])),
                                          "n_top": int(len(top)), "n_bottom": int(len(bot))}
    # incremental: residualize laya_dir on the reaction and item flags (pooled rank regression), then correlate with drift
    flags = [c for c in df.columns if c.startswith("ev_")]
    X = np.column_stack([df["ab_r0"].rank().values] + [df[c].values for c in flags] + [np.ones(len(df))])
    y = df["laya_dir"].rank().values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df["laya_resid"] = y - X @ beta
    res["overall"]["laya_resid"] = {"sp_ab_r_exec": _sp(df["laya_resid"], df["ab_r_exec"]), "sp_ab_r1": _sp(df["laya_resid"], df["ab_r1"])}
    # reaction itself (PEAD-style) for reference
    res["overall"]["reaction_ab_r0"] = {"sp_ab_r_exec": _sp(df["ab_r0"], df["ab_r_exec"])}
    for g in ["ev_202", "ev_502", "ev_801", "ev_101", "ev_701", "ev_neg"]:
        if g in df.columns and (df[g] == 1).sum() >= 100:
            sub = df[df[g] == 1]
            res["by_group"][g] = {"n": int(len(sub)), "sp_dir_ab_r0": _sp(sub["laya_dir"], sub["ab_r0"]), "sp_dir_ab_r_exec": _sp(sub["laya_dir"], sub["ab_r_exec"]),
                                  "sp_resid_ab_r_exec": _sp(sub["laya_resid"], sub["ab_r_exec"]), "sp_reaction_ab_r_exec": _sp(sub["ab_r0"], sub["ab_r_exec"])}
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    L = [f"# {title}\n", f"评分文件 {res['n_scored']:,} 份，其中取到正文并评分 {res['n_with_text']:,} 份，可评估（有收益）{res['n_eval']:,} 份，{res['date_range'][0]} → {res['date_range'][1]}。异常收益 = 个股 − 当日 PIT 等权；ab_r0 信号日反应，ab_r1 次日，ab_r_exec 后 5 日（t+1 进 t+6 出）。\n",
         "## 全部事件：Laya 特征与收益的 Spearman 相关\n", "| 特征 | vs 信号日反应 ab_r0 | vs 次日 ab_r1 | vs 后 5 日 ab_r_exec |\n|---|---|---|---|"]
    for f in feats:
        v = res["overall"][f]; L.append(f"| {f} | {v['sp_ab_r0']:.4f} | {v['sp_ab_r1']:.4f} | {v['sp_ab_r_exec']:.4f} |")
    v = res["overall"]["laya_resid"]; L.append(f"| laya_dir 残差（控制反应与项目编号） | | {v['sp_ab_r1']:.4f} | {v['sp_ab_r_exec']:.4f} |")
    L.append(f"| 反应 ab_r0 本身（PEAD 参照） | | | {res['overall']['reaction_ab_r0']['sp_ab_r_exec']:.4f} |")
    t = res["overall"]["laya_dir_tercile"]
    L.append(f"\nlaya_dir 三分位：最乐观 − 最悲观 的后 5 日异常收益 {t['top_minus_bottom_ab_r_exec']*100:.3f}%（按月聚类 t {t['t_monthly']:.2f}，n {t['n_top']}/{t['n_bottom']}）；信号日反应差 {t['top_minus_bottom_ab_r0']*100:.3f}%。\n")
    L.append("## 分项目\n| 项目 | n | dir vs 反应 | dir vs 后 5 日 | 残差 vs 后 5 日 | 反应 vs 后 5 日 |\n|---|---|---|---|---|---|")
    for g, v in res["by_group"].items():
        L.append(f"| {g} | {v['n']} | {v['sp_dir_ab_r0']:.4f} | {v['sp_dir_ab_r_exec']:.4f} | {v['sp_resid_ab_r_exec']:.4f} | {v['sp_reaction_ab_r_exec']:.4f} |")
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    return res
