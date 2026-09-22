"""Stage 2 pilot: Kronos (K-line foundation model) as a zero-shot signal on the point-in-time S&P 500.

For every scheduled date and every index member with a full lookback window, Kronos-small samples S future
paths of the next `pred_len` daily bars. Path statistics become candidate features:
  kr_ret   mean predicted close at horizon / last close - 1
  kr_med   median version of kr_ret
  kr_pup   share of paths ending above the last close
  kr_vol   dispersion (std of log terminal return) across paths
  kr_ret1  one-step-ahead expected return
  kr_pvol  average within-path daily volatility
The evaluation compares each feature's cross-sectional RankIC against realized forward returns with simple
momentum / volatility features computed on the same panel.

Kronos's pre-training data runs to June 2024 (paper, Sec. D.3.1), so only dates from 2024-07-01 on are clean
out-of-sample; earlier dates are a leakage diagnostic, not a test.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party"))
from kronos_model.kronos import Kronos, KronosTokenizer, calc_time_stamps, sample_from_logits  # noqa: E402

PRICE_COLS = ["open", "high", "low", "close"]
FEAT_COLS = PRICE_COLS + ["volume", "amount"]
KR_FEATURES = ["kr_ret", "kr_med", "kr_pup", "kr_vol", "kr_ret1", "kr_pvol"]
BENCH_FEATURES = ["mom5", "mom20", "mom60", "vol20", "rev1"]


# ----------------------------------------------------------------------------- model
def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_kronos(size: str = "small", device: str | None = None, cache_dir: Path | None = None):
    cache_dir = str(cache_dir or REPO / "data" / "models" / "hf")
    tok_name = "NeoQuasar/Kronos-Tokenizer-2k" if size == "mini" else "NeoQuasar/Kronos-Tokenizer-base"
    tok = KronosTokenizer.from_pretrained(tok_name, cache_dir=cache_dir)
    mdl = Kronos.from_pretrained(f"NeoQuasar/Kronos-{size}", cache_dir=cache_dir)
    device = pick_device(device)
    tok = tok.to(device).eval()
    mdl = mdl.to(device).eval()
    logger.info(f"Kronos-{size} on {device}: {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M params")
    return mdl, tok, device


@torch.no_grad()
def generate_paths(model, tokenizer, x: np.ndarray, x_stamp: np.ndarray, y_stamp: np.ndarray, pred_len: int, *,
                   device: str, max_context: int = 512, clip: float = 5.0, T: float = 1.0, top_k: int = 0,
                   top_p: float = 0.9, sample_count: int = 20) -> np.ndarray:
    """Same procedure as kronos.auto_regressive_inference but returns every sampled path:
    (B, sample_count, pred_len, 6) in the normalized units of `x`."""
    x = torch.from_numpy(x).float().to(device)
    x_stamp = torch.from_numpy(x_stamp).float().to(device)
    y_stamp = torch.from_numpy(y_stamp).float().to(device)
    x = torch.clip(x, -clip, clip)
    B, L = x.size(0), x.size(1)
    x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, L, x.size(2))
    x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, L, x_stamp.size(2))
    y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, pred_len, y_stamp.size(2))
    x_token = tokenizer.encode(x, half=True)
    batch = x_token[0].size(0)
    total = L + pred_len
    full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
    gen_pre = x_token[0].new_empty(batch, pred_len)
    gen_post = x_token[1].new_empty(batch, pred_len)
    pre_buf = x_token[0].new_zeros(batch, max_context)
    post_buf = x_token[1].new_zeros(batch, max_context)
    blen = min(L, max_context)
    s0 = max(0, L - max_context)
    pre_buf[:, :blen] = x_token[0][:, s0 : s0 + blen]
    post_buf[:, :blen] = x_token[1][:, s0 : s0 + blen]
    for i in range(pred_len):
        cur = L + i
        wl = min(cur, max_context)
        toks = [pre_buf[:, :wl], post_buf[:, :wl]] if cur <= max_context else [pre_buf, post_buf]
        cs = max(0, cur - max_context)
        stamp = full_stamp[:, cs:cur, :].contiguous()
        s1_logits, ctx = model.decode_s1(toks[0], toks[1], stamp)
        s_pre = sample_from_logits(s1_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
        s2_logits = model.decode_s2(ctx, s_pre)
        s_post = sample_from_logits(s2_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
        gen_pre[:, i] = s_pre.squeeze(-1)
        gen_post[:, i] = s_post.squeeze(-1)
        if cur < max_context:
            pre_buf[:, cur] = s_pre.squeeze(-1)
            post_buf[:, cur] = s_post.squeeze(-1)
        else:
            pre_buf.copy_(torch.roll(pre_buf, shifts=-1, dims=1))
            post_buf.copy_(torch.roll(post_buf, shifts=-1, dims=1))
            pre_buf[:, -1] = s_pre.squeeze(-1)
            post_buf[:, -1] = s_post.squeeze(-1)
    full_pre = torch.cat([x_token[0], gen_pre], dim=1)
    full_post = torch.cat([x_token[1], gen_post], dim=1)
    cs = max(0, total - max_context)
    z = tokenizer.decode([full_pre[:, cs:total].contiguous(), full_post[:, cs:total].contiguous()], half=True)
    z = z[:, -pred_len:, :].reshape(B, sample_count, pred_len, z.size(2))
    return z.float().cpu().numpy()


# ----------------------------------------------------------------------------- data
def load_panel(norm_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        p = Path(norm_dir) / f"{s}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["date"], usecols=["date"] + PRICE_COLS + ["volume"]).set_index("date")
        out[s] = df
    return out


def load_spells(path: Path) -> pd.DataFrame:
    sp = pd.read_csv(path, parse_dates=["start_eff", "end_eff"])
    return sp[sp["kept"]][["symbol", "start_eff", "end_eff"]]


def members_on(spells: pd.DataFrame, day: pd.Timestamp) -> list[str]:
    m = spells[(spells["start_eff"] <= day) & (spells["end_eff"] >= day)]
    return sorted(set(m["symbol"]))


def schedule(calendar: pd.DatetimeIndex, start: str, end: str, every: int) -> list[pd.Timestamp]:
    cal = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    return list(cal[::every])


def make_window(df: pd.DataFrame, day: pd.Timestamp, lookback: int) -> tuple[np.ndarray, pd.DatetimeIndex] | None:
    if day not in df.index:
        return None
    pos = df.index.get_loc(day)
    if pos < lookback - 1:
        return None
    w = df.iloc[pos - lookback + 1 : pos + 1].copy()
    if pd.isna(w["close"].iloc[-1]):
        return None
    w = w.ffill().bfill()
    if w[PRICE_COLS + ["volume"]].isna().values.any() or (w["close"] <= 0).any():
        return None
    w["amount"] = w["volume"] * w[PRICE_COLS].mean(axis=1)
    return w[FEAT_COLS].values.astype(np.float32), w.index


def normalize(x: np.ndarray, clip: float = 5.0):
    mean, std = x.mean(axis=0), x.std(axis=0)
    xn = np.clip((x - mean) / (std + 1e-5), -clip, clip)
    return xn.astype(np.float32), mean, std


def features_from_paths(paths: np.ndarray, means: np.ndarray, stds: np.ndarray, last_close: np.ndarray) -> pd.DataFrame:
    """paths (B,S,H,6) normalized -> features per series."""
    ci = FEAT_COLS.index("close")
    c = paths[..., ci] * (stds[:, None, None, ci] + 1e-5) + means[:, None, None, ci]  # (B,S,H)
    c = np.clip(c, 1e-6, None)
    c0 = last_close[:, None]
    term = c[:, :, -1] / c0 - 1
    logret = np.log(c[:, :, -1] / c0)
    c0b = np.broadcast_to(c0[:, :, None], (c.shape[0], c.shape[1], 1))
    path_lr = np.diff(np.log(np.concatenate([c0b, c], axis=2)), axis=2)  # (B,S,H) daily log returns along each path
    return pd.DataFrame(
        {
            "kr_ret": term.mean(axis=1),
            "kr_med": np.median(term, axis=1),
            "kr_pup": (term > 0).mean(axis=1),
            "kr_vol": logret.std(axis=1),
            "kr_ret1": (c[:, :, 0] / c0 - 1).mean(axis=1),
            "kr_pvol": path_lr.std(axis=2).mean(axis=1),
        }
    )


# ----------------------------------------------------------------------------- pilot
def run_pilot(*, start: str, end: str, every: int, lookback: int, pred_len: int, sample_count: int, batch_size: int,
              size: str, device: str | None, out_path: Path, norm_dir: Path, spells_path: Path, T: float = 1.0,
              top_p: float = 0.9, checkpoint_every: int = 5) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spells = load_spells(spells_path)
    symbols = sorted(set(spells["symbol"]))
    panel = load_panel(norm_dir, symbols)
    calendar = pd.DatetimeIndex(pd.read_csv(Path(norm_dir) / "SPY.csv", usecols=["date"])["date"].pipe(pd.to_datetime))
    dates = schedule(calendar, start, end, every)
    done_dates: set = set()
    frames: list[pd.DataFrame] = []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        frames.append(prev)
        done_dates = set(pd.to_datetime(prev["date"]).unique())
        logger.info(f"resuming: {len(done_dates)} dates already in {out_path.name}")
    mdl, tok, device = load_kronos(size, device)
    t0 = time.time()
    n_new = 0
    for di, day in enumerate(dates):
        if day in done_dates:
            continue
        mem = members_on(spells, day)
        xs, stamps_x, stamps_y, syms, means, stds, c0s = [], [], [], [], [], [], []
        pos_cal = calendar.get_loc(day)
        future = calendar[pos_cal + 1 : pos_cal + 1 + pred_len]
        if len(future) < pred_len:  # extend past the calendar with business days (stamps only)
            extra = pd.bdate_range(calendar[-1] + pd.Timedelta(days=1), periods=pred_len - len(future))
            future = future.append(extra)
        y_stamp = calc_time_stamps(pd.Series(future)).values.astype(np.float32)
        for s in mem:
            df = panel.get(s)
            if df is None:
                continue
            w = make_window(df, day, lookback)
            if w is None:
                continue
            x, idx = w
            xn, mean, std = normalize(x)
            xs.append(xn)
            stamps_x.append(calc_time_stamps(pd.Series(idx)).values.astype(np.float32))
            stamps_y.append(y_stamp)
            syms.append(s)
            means.append(mean)
            stds.append(std)
            c0s.append(x[-1, FEAT_COLS.index("close")])
        rows = []
        for b in range(0, len(xs), batch_size):
            sl = slice(b, b + batch_size)
            paths = generate_paths(mdl, tok, np.stack(xs[sl]), np.stack(stamps_x[sl]), np.stack(stamps_y[sl]), pred_len,
                                   device=device, T=T, top_p=top_p, sample_count=sample_count)
            f = features_from_paths(paths, np.stack(means[sl]), np.stack(stds[sl]), np.array(c0s[sl]))
            f.insert(0, "symbol", syms[sl])
            rows.append(f)
        if rows:
            day_df = pd.concat(rows, ignore_index=True)
            day_df.insert(1, "date", day)
            frames.append(day_df)
            n_new += 1
        elapsed = time.time() - t0
        logger.info(f"[{di+1}/{len(dates)}] {day.date()}: {len(syms)} series; elapsed {elapsed/60:.1f} min")
        if n_new and n_new % checkpoint_every == 0:
            pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
    logger.info(f"saved {out_path} ({(time.time()-t0)/60:.1f} min)")
    return out_path


# ----------------------------------------------------------------------------- evaluation
def _spearman_by_date(df: pd.DataFrame, feat: str, target: str) -> pd.Series:
    return df.groupby("date").apply(lambda g: g[feat].corr(g[target], method="spearman") if g[feat].notna().sum() > 20 else np.nan).dropna()


def evaluate(features_path: Path, norm_dir: Path, pred_len: int, out_json: Path, out_md: Path, title: str) -> dict:
    feats = pd.read_parquet(features_path)
    feats["date"] = pd.to_datetime(feats["date"])
    panel = load_panel(norm_dir, sorted(feats["symbol"].unique()))
    rows = []
    for s, g in feats.groupby("symbol"):
        df = panel.get(s)
        if df is None:
            continue
        c = df["close"]
        lc = np.log(c)
        pos = c.index.get_indexer(g["date"])
        ok = pos >= 60
        H = pred_len
        def at(k):
            idx = pos + k
            valid = (idx >= 0) & (idx < len(c))
            out = np.full(len(pos), np.nan)
            out[valid] = c.values[idx[valid]]
            return out
        c0, c1, cH, cH1 = at(0), at(1), at(H), at(H + 1)
        r_exec = cH1 / c1 - 1          # Qlib label: enter next close, exit H days later
        r_now = cH / c0 - 1            # what Kronos literally forecasts
        # realized vol over the next H days (log returns t+1..t+H)
        fut = np.stack([at(k) for k in range(0, H + 1)], axis=1)
        rv = np.nanstd(np.diff(np.log(fut), axis=1), axis=1)
        rows.append(pd.DataFrame({
            "symbol": s, "date": g["date"].values, "r_exec": r_exec, "r_now": r_now, "rv": rv,
            "mom5": c0 / at(-5) - 1, "mom20": c0 / at(-20) - 1, "mom60": c0 / at(-60) - 1,
            "vol20": [np.nanstd(np.diff(lc.values[p-20:p+1])) if p >= 20 else np.nan for p in pos],
            "rev1": -(c0 / at(-1) - 1),
        }))
    tgt = pd.concat(rows, ignore_index=True)
    df = feats.merge(tgt, on=["symbol", "date"], how="left")
    df = df.dropna(subset=["r_exec"])
    res = {"title": title, "n_obs": int(len(df)), "n_dates": int(df["date"].nunique()), "date_range": [str(df["date"].min().date()), str(df["date"].max().date())], "signals": {}}
    for feat in KR_FEATURES + BENCH_FEATURES:
        for target in ["r_exec", "r_now"]:
            ic = _spearman_by_date(df, feat, target)
            res["signals"][f"{feat}|{target}"] = {"rank_ic": float(ic.mean()), "std": float(ic.std()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))), "n": int(len(ic)), "pos": float((ic > 0).mean())}
    for feat in ["kr_vol", "kr_pvol", "vol20"]:
        ic = _spearman_by_date(df, feat, "rv")
        res["signals"][f"{feat}|rv"] = {"rank_ic": float(ic.mean()), "std": float(ic.std()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))), "n": int(len(ic)), "pos": float((ic > 0).mean())}
    # quintile spread of kr_ret on the executable return
    def q_spread(g):
        if g["kr_ret"].notna().sum() < 50:
            return np.nan
        q = pd.qcut(g["kr_ret"].rank(method="first"), 5, labels=False)
        return g.loc[q == 4, "r_exec"].mean() - g.loc[q == 0, "r_exec"].mean()
    qs = df.groupby("date").apply(q_spread).dropna()
    res["kr_ret_q5_minus_q1"] = {"mean_per_period": float(qs.mean()), "t": float(qs.mean() / qs.std() * np.sqrt(len(qs))), "n": int(len(qs)), "annualized_approx": float(qs.mean() * 252 / pred_len)}
    # by year for the headline signal
    ic = _spearman_by_date(df, "kr_ret", "r_exec")
    res["kr_ret_by_year"] = {int(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()}
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    # markdown
    L = [f"# {title}\n", f"样本：{res['n_obs']:,} 个（股票, 日期）对，{res['n_dates']} 个调度日，{res['date_range'][0]} → {res['date_range'][1]}；目标 r_exec = t+1 收盘进、t+{pred_len+1} 收盘出（与 Qlib 标签一致），r_now = t → t+{pred_len}。\n",
         "| 信号 | 目标 | RankIC | t 值 | 正天数占比 | n |\n|---|---|---|---|---|---|"]
    for k, v in res["signals"].items():
        f, t = k.split("|")
        L.append(f"| {f} | {t} | {v['rank_ic']:.4f} | {v['t']:.2f} | {v['pos']*100:.0f}% | {v['n']} |")
    q = res["kr_ret_q5_minus_q1"]
    L.append(f"\nkr_ret 五分位多空（Q5−Q1）每期均值 {q['mean_per_period']*100:.3f}%，t {q['t']:.2f}，粗略年化 {q['annualized_approx']*100:.1f}%（不含成本）。\n")
    L.append("kr_ret 对 r_exec 的 RankIC 按年：" + ", ".join(f"{y}: {v:.4f}" for y, v in res["kr_ret_by_year"].items()) + "\n")
    out_md.write_text("\n".join(L), encoding="utf-8")
    return res
