import numpy as np
import pandas as pd
import pytest

from cryptobt.data import clean_prices, load_prices, parse_coinmetrics_csv, save_snapshot

RAW = """time,AdrActCnt,PriceUSD,HashRate
2020-01-01,10,100.0,1
2020-01-02,10,101.0,1
2020-01-02,10,101.5,1
2020-01-04,10,,1
2020-01-05,10,-1,1
2020-01-06,10,104.0,1
"""


def test_parse_keeps_only_price():
    s = parse_coinmetrics_csv(RAW)
    assert s.name == "close"
    assert s.index.name == "date"
    assert len(s) == 6


def test_clean_fills_short_gaps_and_drops_bad_prices():
    s, rep = clean_prices(parse_coinmetrics_csv(RAW), "x")
    # duplicate 01-02 keeps the last value
    assert s.loc["2020-01-02"] == 101.5
    # 03, 04, 05 are missing/bad -> forward-filled with 101.5 (3-day gap)
    assert list(s.loc["2020-01-03":"2020-01-05"]) == [101.5] * 3
    assert rep.non_positive_dropped == 1
    assert rep.gaps_filled == 3
    assert s.index.freq is None or s.index.inferred_freq == "D"


def test_clean_leaves_long_gaps_unfilled():
    idx = pd.to_datetime(["2020-01-01", "2020-01-10", "2020-01-11"])
    s, _ = clean_prices(pd.Series([1.0, 2.0, 3.0], index=idx), "x", max_gap=2)
    assert s.isna().sum() == 0  # NaNs dropped
    assert pd.Timestamp("2020-01-05") not in s.index


def test_snapshot_roundtrip(tmp_path):
    idx = pd.date_range("2016-01-01", periods=5, freq="D", name="date")
    s = pd.Series(np.arange(1.0, 6.0), index=idx, name="close")
    save_snapshot("btc", s, tmp_path)
    df = load_prices(["btc"], start="2016-01-02", data_dir=tmp_path)
    assert list(df["btc"]) == [2.0, 3.0, 4.0, 5.0]


def test_missing_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices(["doge"], data_dir=tmp_path)
