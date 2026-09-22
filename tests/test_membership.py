import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ljs.membership import build_spells, filter_spells_by_data, norm_ticker, classify_reason  # noqa: E402

START, END = "2010-01-01", "2026-09-21"


def _universe(rows):
    return pd.DataFrame([{"yahoo": s, "date_added": d} for s, d in rows])


def _changes(rows):
    return pd.DataFrame([{"date": pd.Timestamp(d), "added": a, "added_name": None, "removed": r, "removed_name": None, "reason": None} for d, a, r in rows])


def test_norm_ticker():
    assert norm_ticker("BRK.B[1]") == "BRK-B" and norm_ticker(float("nan")) is None


def test_reason():
    assert classify_reason("Acquired by X") == "acquisition" and classify_reason("Market capitalization changes.") == "demotion"


def test_spells_rules():
    uni = _universe([("CUR", "1995-03-01"), ("NEW", "2015-06-01"), ("META", "2013-12-23"), ("BACK", "2018-01-01")])
    ch = _changes([
        ("2015-06-01", "NEW", "OLD"),        # OLD removed w/o addition -> since window start; NEW added -> current
        ("2013-12-23", "FB", "X1"),          # FB added, later renamed META: never removed, not current -> dropped
        ("2012-01-01", "BACK", None),        # BACK added 2012, removed 2014, current again (re-added w/o event)
        ("2014-01-01", None, "BACK"),
        ("2011-05-05", None, "X1"),          # X1 removed 2011 (also appears removed 2013 -> second spell since... treated: remove with no open -> since start again)
    ])
    sp, diag = build_spells(uni, ch, START, END)
    g = {s: df for s, df in sp.groupby("symbol")}
    assert list(g["CUR"][["start", "end"]].iloc[0]) == [pd.Timestamp(START), pd.Timestamp(END)]      # date_added < window -> clamped
    assert list(g["NEW"][["start", "end"]].iloc[0]) == [pd.Timestamp("2015-06-01"), pd.Timestamp(END)]
    assert list(g["OLD"][["start", "end"]].iloc[0]) == [pd.Timestamp(START), pd.Timestamp("2015-05-31")]
    assert "FB" not in g and "FB" in diag["renamed_or_missing_removal"]
    assert list(g["META"][["start", "end"]].iloc[0]) == [pd.Timestamp("2013-12-23"), pd.Timestamp(END)]
    back = g["BACK"].reset_index(drop=True)
    assert len(back) == 2 and back.loc[0, "end"] == pd.Timestamp("2013-12-31") and back.loc[1, "start"] == pd.Timestamp("2018-01-01")
    assert "BACK" in diag["readded_without_event"] and "OLD" in diag["removed_without_addition"]


def test_filter_by_data_reuse():
    sp = pd.DataFrame([
        {"symbol": "FB", "start": pd.Timestamp("2013-12-23"), "end": pd.Timestamp("2022-06-08"), "source": "events"},
        {"symbol": "OK", "start": pd.Timestamp("2010-01-01"), "end": pd.Timestamp("2015-05-31"), "source": "events"},
        {"symbol": "NO", "start": pd.Timestamp("2010-01-01"), "end": pd.Timestamp("2012-05-31"), "source": "events"},
    ])
    ranges = {"FB": (pd.Timestamp("2025-06-26"), pd.Timestamp("2026-09-21")),   # recycled ticker
              "OK": (pd.Timestamp("2010-06-01"), pd.Timestamp("2026-09-21"))}   # data starts inside spell -> clamped
    out = filter_spells_by_data(sp, ranges)
    r = out.set_index("symbol")
    assert not r.loc["FB", "kept"] and r.loc["FB", "why"] == "ticker_reuse_or_late_data"
    assert r.loc["OK", "kept"] and r.loc["OK", "why"] == "clamped" and r.loc["OK", "start_eff"] == pd.Timestamp("2010-06-01")
    assert not r.loc["NO", "kept"] and r.loc["NO", "why"] == "no_data"
