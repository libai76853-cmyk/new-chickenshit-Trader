"""Stage 2 pilot: Kronos zero-shot signal on the point-in-time universe.

Generate:  .venv/bin/python scripts/03_kronos_pilot.py generate --start 2024-07-01 --end 2026-09-21 --every 5 --lookback 90 --pred-len 5 --samples 20 --batch 64 --size small --out data/kronos/pilot_post2024h2.parquet
Evaluate:  .venv/bin/python scripts/03_kronos_pilot.py evaluate --features data/kronos/pilot_post2024h2.parquet --pred-len 5 --tag post2024h2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--start", required=True); g.add_argument("--end", required=True)
    g.add_argument("--every", type=int, default=5); g.add_argument("--lookback", type=int, default=90)
    g.add_argument("--pred-len", type=int, default=5); g.add_argument("--samples", type=int, default=20)
    g.add_argument("--batch", type=int, default=64); g.add_argument("--size", default="small")
    g.add_argument("--device", default=None); g.add_argument("--T", type=float, default=1.0); g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--out", required=True)
    g.add_argument("--spells", default=str(REPO / "data" / "membership_spells.csv"))
    e = sub.add_parser("evaluate")
    e.add_argument("--features", required=True); e.add_argument("--pred-len", type=int, default=5); e.add_argument("--tag", required=True)
    e.add_argument("--title", default=None)
    args = ap.parse_args()
    from ljs.kronos_features import run_pilot, evaluate

    norm_dir = REPO / "data" / "normalized"
    if args.cmd == "generate":
        run_pilot(start=args.start, end=args.end, every=args.every, lookback=args.lookback, pred_len=args.pred_len,
                  sample_count=args.samples, batch_size=args.batch, size=args.size, device=args.device,
                  out_path=REPO / args.out, norm_dir=norm_dir, spells_path=Path(args.spells), T=args.T, top_p=args.top_p)
    else:
        out_json = REPO / "reports" / f"kronos_{args.tag}.json"
        out_md = REPO / "reports" / f"kronos_{args.tag}.md"
        res = evaluate(REPO / args.features, norm_dir, args.pred_len, out_json, out_md, args.title or f"Kronos 零样本信号评估（{args.tag}）")
        print(f"REPORT {out_md}")
        k = res["signals"]["kr_ret|r_exec"]
        print(f"kr_ret vs r_exec: RankIC {k['rank_ic']:.4f} t {k['t']:.2f} n {k['n']}")


if __name__ == "__main__":
    main()
