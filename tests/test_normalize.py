"""Invariants of the Qlib-style Yahoo normalization."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ljs.data_yahoo import normalize_1d  # noqa: E402
from ljs.universe import to_yahoo_symbol  # noqa: E402


def _synthetic_raw() -> pd.DataFrame:
    """20 business days; 2:1 split on day 10 (raw price halves, adjclose continuous); zero-volume day 5."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    adj = 100 + np.arange(20, dtype=float)  # smooth adjusted close
    raw_close = adj.copy()
    raw_close[:10] *= 2  # before the split, raw prices were twice as high
    df = pd.DataFrame(
        {
            "date": dates,
            "open": raw_close * 0.99,
            "high": raw_close * 1.01,
            "low": raw_close * 0.98,
            "close": raw_close,
            "adjclose": adj,
            "volume": np.where(np.arange(20) < 10, 1_000.0, 2_000.0),
        }
    )
    df.loc[5, "volume"] = 0  # non-trading day
    return df


def test_symbol_mapping():
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert to_yahoo_symbol(" aapl ") == "AAPL"


def test_normalize_invariants():
    raw = _synthetic_raw()
    cal = pd.DatetimeIndex(raw["date"])
    out = normalize_1d(raw, "TEST", cal)
    assert list(out.columns) == ["symbol", "date", "open", "high", "low", "close", "volume", "factor", "change"]
    assert len(out) == 20
    out = out.set_index("date")
    good = out["close"].notna()
    # zero-volume day is blanked
    assert not good.loc[raw.loc[5, "date"]]
    # normalized close is continuous across the split: equals adjclose scaled by first close
    adj = raw.set_index("date")["adjclose"]
    expected = adj / adj.iloc[0]
    np.testing.assert_allclose(out.loc[good, "close"], expected[good], rtol=1e-9)
    # Qlib convention: raw price == $close / $factor
    raw_close = raw.set_index("date")["close"]
    np.testing.assert_allclose(out.loc[good, "close"] / out.loc[good, "factor"], raw_close[good], rtol=1e-9)
    # change is computed on raw close (so it shows the split jump, as Qlib does)
    c = raw_close.copy(); c.loc[raw.loc[5, "date"]] = np.nan; c = c.ffill()
    np.testing.assert_allclose(out.loc[good, "change"].iloc[1:], (c / c.shift(1) - 1)[good].iloc[1:], rtol=1e-9)
    # dollar volume is preserved by the (factor, first-close) scaling
    raw_vol = raw.set_index("date")["volume"]
    ratio = (out.loc[good, "volume"] * out.loc[good, "close"]) / (raw_vol[good] * raw_close[good])
    np.testing.assert_allclose(ratio, ratio.iloc[0], rtol=1e-9)
