"""Stage-1 baseline: Qlib Alpha158 features + LightGBM -> IC/RankIC on the test period ->
TopkDropout long-only backtest with costs vs SPY -> markdown report + plot.

Run via scripts/02_run_baseline.py. Everything is driven by configs/baseline_lgb.yaml.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # mlflow>=3.16 refuses ./mlruns otherwise; qlib logs there
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import yaml
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
TRADING_DAYS = 252


# ----------------------------------------------------------------------------- config / init
def load_cfg(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def init_qlib(cfg: dict) -> None:
    import qlib
    from qlib.constant import REG_US, REG_CN

    region = REG_US if cfg["qlib"].get("region", "us") == "us" else REG_CN
    qlib.init(
        provider_uri=str(REPO / cfg["data"]["qlib_dir"]),
        region=region,
        kernels=cfg["qlib"].get("kernels", 8),
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": f"file:{REPO / 'mlruns'}", "default_exp_name": "baseline"},
        },
    )


def data_end(cfg: dict) -> str:
    cal = (REPO / cfg["data"]["qlib_dir"] / "calendars" / "day.txt").read_text().split()
    return cal[-1]


def resolve_segments(cfg: dict, end: str) -> dict[str, tuple[str, str]]:
    segs = {}
    for k, (a, b) in cfg["dataset"]["segments"].items():
        segs[k] = (a, b or end)
    return segs


# ----------------------------------------------------------------------------- dataset / model
def build_dataset(cfg: dict, segments: dict):
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    dcfg = cfg["dataset"]
    train_start, train_end = segments["train"]
    handler = Alpha158(
        instruments=cfg["data"]["universe_name"],
        start_time=train_start,
        end_time=segments["test"][1],
        fit_start_time=train_start,
        fit_end_time=train_end,
        infer_processors=dcfg["infer_processors"],
        learn_processors=dcfg["learn_processors"],
        label=[dcfg["label"]],
    )
    return DatasetH(handler, segments=segments)


def train_model(cfg: dict, ds):
    from qlib.contrib.model.gbdt import LGBModel

    m = cfg["model"]
    model = LGBModel(
        early_stopping_rounds=m["early_stopping_rounds"], num_boost_round=m["num_boost_round"], **m["kwargs"]
    )
    model.fit(ds, verbose_eval=50)
    return model


# ----------------------------------------------------------------------------- evaluation
def ic_stats(pred: pd.Series, label: pd.Series) -> tuple[dict, pd.DataFrame]:
    from qlib.contrib.eva.alpha import calc_ic

    ic, ric = calc_ic(pred, label)
    daily = pd.DataFrame({"ic": ic, "rank_ic": ric}).dropna()
    stats = {
        "n_days": int(len(daily)),
        "ic_mean": float(daily["ic"].mean()),
        "ic_std": float(daily["ic"].std()),
        "icir": float(daily["ic"].mean() / daily["ic"].std()),
        "rank_ic_mean": float(daily["rank_ic"].mean()),
        "rank_ic_std": float(daily["rank_ic"].std()),
        "rank_icir": float(daily["rank_ic"].mean() / daily["rank_ic"].std()),
        "rank_ic_pos_ratio": float((daily["rank_ic"] > 0).mean()),
        # t-stat of mean RankIC assuming iid days (optimistic for a 5-day label; overlapping horizons)
        "rank_ic_tstat": float(daily["rank_ic"].mean() / daily["rank_ic"].std() * np.sqrt(len(daily))),
    }
    return stats, daily


def ic_by_year(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby(daily.index.year)
    out = pd.DataFrame({"ic": g["ic"].mean(), "rank_ic": g["rank_ic"].mean(), "rank_icir": g["rank_ic"].mean() / g["rank_ic"].std(), "days": g.size()})
    out.index.name = "year"
    return out


def run_backtest(cfg: dict, pred: pd.Series, start: str, end: str):
    from qlib.backtest import backtest
    from qlib.contrib.strategy import TopkDropoutStrategy

    b = cfg["backtest"]
    strategy = TopkDropoutStrategy(signal=pred, topk=b["topk"], n_drop=b["n_drop"])
    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    exchange_kwargs = {
        "freq": "day",
        "limit_threshold": b.get("limit_threshold"),
        "deal_price": b.get("deal_price", "close"),
        "open_cost": b["open_cost"],
        "close_cost": b["close_cost"],
        "min_cost": b["min_cost"],
    }
    pm, _ind = backtest(
        start_time=start, end_time=end, strategy=strategy, executor=executor_cfg,
        benchmark=cfg["data"]["benchmark"], account=b["account"], exchange_kwargs=exchange_kwargs,
    )
    report, positions = pm["1day"]
    return report, positions


def perf_stats(r: pd.Series) -> dict:
    """Annualized stats of a daily return series (simple sum convention, like qlib.risk_analysis with N=252)."""
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe_or_ir": float(ann_ret / ann_vol) if ann_vol > 0 else float("nan"),
        "max_drawdown": float(dd.min()),
        "total_return": float(cum.iloc[-1] - 1),
        "n_days": int(len(r)),
    }


def yearly_table(report: pd.DataFrame) -> pd.DataFrame:
    strat = report["return"] - report["cost"]
    df = pd.DataFrame({"strategy": strat, "spy": report["bench"], "excess": strat - report["bench"], "turnover": report["turnover"]})
    g = df.groupby(df.index.year)
    out = pd.DataFrame(
        {
            "strategy": g["strategy"].apply(lambda x: (1 + x).prod() - 1),
            "spy": g["spy"].apply(lambda x: (1 + x).prod() - 1),
            "excess": g["excess"].sum(),
            "avg_daily_turnover": g["turnover"].mean(),
            "days": g.size(),
        }
    )
    out.index.name = "year"
    return out


def feature_importance(model, n: int = 20) -> pd.Series:
    fi = model.get_feature_importance(importance_type="gain")
    return fi.sort_values(ascending=False).head(n)


# ----------------------------------------------------------------------------- report
def plot_curves(report: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strat = (1 + report["return"] - report["cost"]).cumprod()
    bench = (1 + report["bench"]).cumprod()
    excess = (1 + report["return"] - report["cost"] - report["bench"]).cumprod()
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(strat.index, strat.values, label="LightGBM top-k (after cost)")
    ax[0].plot(bench.index, bench.values, label="SPY")
    ax[0].set_ylabel("growth of 1")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(excess.index, excess.values, color="tab:green", label="cumulative excess (after cost)")
    ax[1].axhline(1, color="grey", lw=0.8)
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.suptitle("Baseline: Alpha158 + LightGBM, S&P 500, daily rebalance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _pct(x: float) -> str:
    return f"{x*100:.2f}%"


def write_report(path: Path, cfg: dict, segments: dict, meta: dict, ic: dict, ic_year: pd.DataFrame,
                 perf: dict, yearly: pd.DataFrame, fi: pd.Series, plot_name: str, best_iter: int) -> None:
    d = cfg["dataset"]; b = cfg["backtest"]; m = cfg["model"]["kwargs"]
    L = []
    L.append(f"# 基线报告：Alpha158 + LightGBM（标普 500，日频）\n")
    L.append(f"生成时间：{meta['generated_at']}  ·  代码版本：{meta.get('git_rev','n/a')}\n")
    L.append("## 数据\n")
    L.append(f"- 股票池：当前标普 500 成分股 {meta['n_instruments']} 只（Wikipedia 名单，存在幸存者偏差，见下文）")
    L.append(f"- 基准：{cfg['data']['benchmark']}；数据：Yahoo 日线，{meta['data_start']} → {meta['data_end']}，{meta['n_days_total']} 个交易日")
    L.append(f"- 特征：Qlib Alpha158（{meta['n_features']} 个），RobustZScore 归一化 + 缺失填零；标签：`{d['label']}`（{d['horizon_days']} 日前向收益，t+1 收盘进、t+{d['horizon_days']+1} 收盘出，无未来函数），训练时对标签做横截面排序归一化")
    L.append(f"- 切分：训练 {segments['train'][0]} → {segments['train'][1]}；验证 {segments['valid'][0]} → {segments['valid'][1]}；测试 {segments['test'][0]} → {segments['test'][1]}（单次切分，非滚动）\n")
    L.append("## 模型\n")
    L.append(f"- LightGBM，Qlib 公开基准参数：learning_rate {m['learning_rate']}，num_leaves {m['num_leaves']}，max_depth {m['max_depth']}，lambda_l1 {m['lambda_l1']}，lambda_l2 {m['lambda_l2']}，colsample {m['colsample_bytree']}，subsample {m['subsample']}")
    L.append(f"- 早停：验证集 {cfg['model']['early_stopping_rounds']} 轮无提升即停；实际最佳轮数 {best_iter}\n")
    L.append("## 预测力（测试期，样本外）\n")
    L.append("| 指标 | 值 |\n|---|---|")
    L.append(f"| 交易日数 | {ic['n_days']} |")
    L.append(f"| IC 均值 | {ic['ic_mean']:.4f} |")
    L.append(f"| ICIR | {ic['icir']:.3f} |")
    L.append(f"| RankIC 均值 | {ic['rank_ic_mean']:.4f} |")
    L.append(f"| RankIC 标准差 | {ic['rank_ic_std']:.4f} |")
    L.append(f"| RankICIR | {ic['rank_icir']:.3f} |")
    L.append(f"| RankIC>0 的天数占比 | {_pct(ic['rank_ic_pos_ratio'])} |")
    L.append(f"| RankIC t 统计量（按日独立假设，偏乐观） | {ic['rank_ic_tstat']:.2f} |\n")
    L.append("按年：\n")
    L.append("| 年 | IC | RankIC | RankICIR | 天数 |\n|---|---|---|---|---|")
    for y, r in ic_year.iterrows():
        L.append(f"| {y} | {r['ic']:.4f} | {r['rank_ic']:.4f} | {r['rank_icir']:.3f} | {int(r['days'])} |")
    L.append("")
    L.append("## 回测（多头 top-k，等权，日频调仓，扣费）\n")
    L.append(f"- 策略：每日按预测分数持有前 {b['topk']} 只，每日最多换出 {b['n_drop']} 只（Qlib TopkDropout）；初始资金 {b['account']:,}；成本买卖各 {b['open_cost']*1e4:.0f} bp，最低 {b['min_cost']} 美元；收盘价成交\n")
    L.append("| 指标 | 策略（扣费） | SPY | 超额（扣费） |\n|---|---|---|---|")
    s, k, e = perf["strategy"], perf["spy"], perf["excess"]
    L.append(f"| 年化收益 | {_pct(s['ann_return'])} | {_pct(k['ann_return'])} | {_pct(e['ann_return'])} |")
    L.append(f"| 年化波动 | {_pct(s['ann_vol'])} | {_pct(k['ann_vol'])} | {_pct(e['ann_vol'])} |")
    L.append(f"| 夏普 / 信息比率 | {s['sharpe_or_ir']:.2f} | {k['sharpe_or_ir']:.2f} | {e['sharpe_or_ir']:.2f} |")
    L.append(f"| 最大回撤 | {_pct(s['max_drawdown'])} | {_pct(k['max_drawdown'])} | {_pct(e['max_drawdown'])} |")
    L.append(f"| 区间总收益 | {_pct(s['total_return'])} | {_pct(k['total_return'])} | {_pct(e['total_return'])} |")
    L.append(f"| 扣费前超额年化 | {_pct(perf['excess_gross']['ann_return'])} | | 成本拖累年化 {_pct(perf['excess_gross']['ann_return'] - e['ann_return'])} |\n")
    L.append("按年：\n")
    L.append("| 年 | 策略 | SPY | 超额 | 日均换手 | 天数 |\n|---|---|---|---|---|---|")
    for y, r in yearly.iterrows():
        L.append(f"| {y} | {_pct(r['strategy'])} | {_pct(r['spy'])} | {_pct(r['excess'])} | {_pct(r['avg_daily_turnover'])} | {int(r['days'])} |")
    L.append("")
    L.append(f"![curves]({plot_name})\n")
    L.append("## 最重要的 20 个特征（LightGBM gain）\n")
    L.append("| 特征 | gain 占比 |\n|---|---|")
    tot = fi.sum()
    for name, v in fi.items():
        L.append(f"| {name} | {_pct(v / tot)} |")
    L.append("")
    L.append("## 必须知道的局限\n")
    L.append("1. **幸存者偏差**：股票池是 2026 年的标普 500 名单，回测期内被剔除、破产、被收购的公司不在里面。这会抬高多头组合收益，也可能抬高 IC。修复需要历史成分股名单（下一阶段）。")
    L.append("2. **单次切分**：一次训练、一段测试。生产上要滚动重训（每年或每季），结果通常会打折。")
    L.append("3. **成本假设简化**：每边 5 bp 是大盘股的粗估，未建模冲击成本和隔夜跳空；日频调仓的换手率决定了成本敏感度，见按年表的换手列。")
    L.append("4. **标签与调仓错配**：标签是 5 日收益，回测是日频 TopkDropout 调仓，持仓期由 n_drop 隐式决定，不是严格的 5 日持有。")
    L.append("5. **公开因子、公开模型**：Alpha158 + LightGBM 是所有人都能跑的组合，任何超额都应默认是拥挤的。")
    L.append("")
    L.append("## 下一步（按 docs/02 方案）\n")
    L.append("- 记录本报告的 RankIC / 超额作为基线数字；后续每加一层（Kronos 特征、Laya 事件特征）都只看相对这里的增量。")
    L.append("- 滚动训练版本；历史成分股名单；再决定是否上 IBKR 模拟盘。")
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def run(cfg_path: str | Path, tag: str | None = None) -> Path:
    cfg = load_cfg(cfg_path)
    t0 = dt.datetime.now()
    qdir = REPO / cfg["data"]["qlib_dir"]
    if not (qdir / "calendars" / "day_future.txt").exists():
        from ljs.dump_qlib import write_future_calendar
        write_future_calendar(qdir)
    init_qlib(cfg)
    end = data_end(cfg)
    segments = resolve_segments(cfg, end)
    logger.info(f"segments: {segments}")

    ds = build_dataset(cfg, segments)
    train_df = ds.prepare("train", col_set=["feature", "label"], data_key="learn")
    logger.info(f"train rows {len(train_df):,}, features {train_df['feature'].shape[1]}")
    n_features = train_df["feature"].shape[1]
    del train_df

    model = train_model(cfg, ds)
    best_iter = int(getattr(model.model, "best_iteration", 0) or 0)
    logger.info(f"trained; best_iteration={best_iter}")

    pred = model.predict(ds, segment="test")
    label = ds.prepare("test", col_set="label", data_key="infer").iloc[:, 0]
    pred, label = pred.align(label, join="inner")
    ic, ic_daily = ic_stats(pred, label)
    logger.info(f"test IC {ic['ic_mean']:.4f} RankIC {ic['rank_ic_mean']:.4f} RankICIR {ic['rank_icir']:.3f}")

    report, positions = run_backtest(cfg, pred, segments["test"][0], segments["test"][1])
    strat_net = report["return"] - report["cost"]
    perf = {
        "strategy": perf_stats(strat_net),
        "spy": perf_stats(report["bench"]),
        "excess": perf_stats(strat_net - report["bench"]),
        "excess_gross": perf_stats(report["return"] - report["bench"]),
    }
    logger.info(f"backtest: strat ann {perf['strategy']['ann_return']:.3f} vs SPY {perf['spy']['ann_return']:.3f}; excess IR {perf['excess']['sharpe_or_ir']:.2f}")

    # ---- persist
    stamp = tag or dt.date.today().strftime("%Y%m%d")
    out_dir = REPO / "reports"
    out_dir.mkdir(exist_ok=True)
    art_dir = REPO / "data" / "artifacts" / f"baseline_lgb_{stamp}"
    art_dir.mkdir(parents=True, exist_ok=True)
    pred.rename("score").to_frame().join(label.rename("label")).to_parquet(art_dir / "pred_test.parquet")
    report.to_parquet(art_dir / "backtest_report.parquet")
    ic_daily.to_parquet(art_dir / "ic_daily.parquet")
    model.model.save_model(str(art_dir / "lgb_model.txt"))

    cal = (REPO / cfg["data"]["qlib_dir"] / "calendars" / "day.txt").read_text().split()
    n_inst = len((REPO / cfg["data"]["qlib_dir"] / "instruments" / f"{cfg['data']['universe_name']}.txt").read_text().splitlines())
    meta = {
        "generated_at": t0.strftime("%Y-%m-%d %H:%M"),
        "git_rev": os.popen(f"git -C '{REPO}' rev-parse --short HEAD 2>/dev/null").read().strip() or "n/a",
        "n_instruments": n_inst,
        "data_start": cal[0],
        "data_end": cal[-1],
        "n_days_total": len(cal),
        "n_features": n_features,
    }
    plot_name = f"baseline_lgb_{stamp}.png"
    plot_curves(report, out_dir / plot_name)
    fi = feature_importance(model, 20)
    yearly = yearly_table(report)
    report_path = out_dir / f"baseline_lgb_{stamp}.md"
    write_report(report_path, cfg, segments, meta, ic, ic_by_year(ic_daily), perf, yearly, fi, plot_name, best_iter)
    metrics = {"meta": meta, "segments": segments, "ic": ic, "perf": perf, "best_iteration": best_iter,
               "ic_by_year": ic_by_year(ic_daily).reset_index().to_dict(orient="records"),
               "yearly": yearly.reset_index().to_dict(orient="records"), "top_features": fi.to_dict()}
    (out_dir / f"baseline_lgb_{stamp}.json").write_text(json.dumps(metrics, indent=1, default=float), encoding="utf-8")
    logger.info(f"report -> {report_path} ({(dt.datetime.now()-t0).total_seconds()/60:.1f} min)")
    return report_path
