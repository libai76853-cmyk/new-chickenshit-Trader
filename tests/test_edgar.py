import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ljs.edgar import signal_dates, build_events  # noqa: E402

CAL = pd.DatetimeIndex(pd.bdate_range("2024-07-22", "2024-08-02"))  # Mon-Fri, no holidays in range


def test_signal_dates_eastern():
    acc = pd.Series(["2024-07-25T09:15:00.000Z",   # before close on a trading day -> same day
                     "2024-07-25T16:05:00.000Z",   # after close -> next trading day
                     "2024-07-27T10:00:00.000Z",   # Saturday -> Monday
                     "2024-07-26T15:59:59.000Z"])  # just before close Friday -> Friday
    out = signal_dates(acc, CAL, tz="US/Eastern")
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2024-07-25", "2024-07-26", "2024-07-29", "2024-07-26"]


def test_signal_dates_utc_reading():
    acc = pd.Series(["2024-07-25T20:05:00.000Z"])  # 16:05 ET if the Z were real UTC -> next day
    out = signal_dates(acc, CAL, tz="UTC")
    assert out.iloc[0].strftime("%Y-%m-%d") == "2024-07-26"


def test_build_events_flags():
    filings = pd.DataFrame({
        "accessionNumber": ["0001-24-1", "0001-24-2", "0001-24-3"],
        "filingDate": ["2024-07-25", "2024-07-26", "2024-07-26"],
        "reportDate": ["2024-07-25", "2024-07-26", "2024-07-26"],
        "acceptanceDateTime": ["2024-07-25T16:30:00.000Z", "2024-07-26T08:00:00.000Z", "2024-07-26T08:00:00.000Z"],
        "form": ["8-K", "8-K/A", "10-Q"],
        "items": ["2.02,9.01", "4.02,5.02", None],
        "primaryDocument": ["a.htm", "b.htm", "c.htm"], "primaryDocDescription": ["", "", ""], "size": [1, 1, 1],
        "cik": [320193, 320193, 320193],
    })
    ev = build_events(filings, {"AAPL": 320193}, CAL, start="2024-01-01")
    assert len(ev) == 2 and set(ev["symbol"]) == {"AAPL"}
    e1 = ev[ev["accessionNumber"] == "0001-24-1"].iloc[0]
    assert e1["ev_202"] == 1 and e1["ev_neg"] == 0 and e1["signal_date"] == pd.Timestamp("2024-07-26")  # after close -> next day
    e2 = ev[ev["accessionNumber"] == "0001-24-2"].iloc[0]
    assert e2["ev_402"] == 1 and e2["ev_502"] == 1 and e2["ev_neg"] == 1 and e2["is_amend"] == 1 and e2["signal_date"] == pd.Timestamp("2024-07-26")
