import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ljs.laya_events import features_from_answers, select_events, QUESTIONS  # noqa: E402


def test_features_from_answers():
    ans = {
        "direction": {"probabilities": {"positive": 0.6, "negative": 0.1, "neutral": 0.3}},
        "guidance": {"probabilities": {"raised": 0.5, "lowered": 0.1, "maintained": 0.2, "none": 0.2}},
        "results": {"probabilities": {"beat": 0.7, "miss": 0.1, "inline": 0.1, "none": 0.1}},
        "exec_change": {"noul": 0.05}, "mna": {"noul": 0.2}, "restatement": {"noul": 0.01},
        "importance": {"score": 2.4},
    }
    f = features_from_answers(ans)
    assert abs(f["laya_dir"] - 0.5) < 1e-9 and abs(f["guid_net"] - 0.4) < 1e-9 and abs(f["res_net"] - 0.6) < 1e-9
    assert f["p_exec"] == 0.05 and f["importance"] == 2.4
    assert set(QUESTIONS) == {"direction", "guidance", "results", "exec_change", "mna", "restatement", "importance"}


def test_select_events_membership_and_groups():
    events = pd.DataFrame({
        "accessionNumber": ["a1", "a2", "a3", "a4"],
        "symbol": ["AAA", "AAA", "BBB", "AAA"],
        "signal_date": pd.to_datetime(["2024-08-01", "2024-08-02", "2024-08-01", "2023-01-05"]),
        "ev_202": [1, 0, 1, 1], "ev_502": [0, 1, 0, 0], "ev_801": [0, 0, 0, 0],
    })
    spells = pd.DataFrame({"symbol": ["AAA", "BBB"], "start_eff": pd.to_datetime(["2020-01-01", "2024-09-01"]), "end_eff": pd.to_datetime(["2026-12-31", "2026-12-31"])})
    out = select_events(events, "2024-07-01", "2024-12-31", ["ev_202", "ev_502"], spells)
    assert list(out["accessionNumber"]) == ["a1", "a2"]  # a3: BBB not yet a member; a4: outside window
