import numpy as np
import pandas as pd
import pytest

from cryptobt.signals import FAMILIES, buy_and_hold, ma_crossover, no_trade_band, tsmom, vol_target


@pytest.fixture
def close():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-01", periods=600, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))), index=idx)


@pytest.mark.parametrize(
    "fn",
    [
        lambda c: tsmom(c, 30),
        lambda c: tsmom(c, 30, allow_short=True),
        lambda c: ma_crossover(c, 10, 50),
        lambda c: vol_target(tsmom(c, 30), c, 0.4, 20),
        lambda c: no_trade_band(vol_target(tsmom(c, 30), c, 0.4, 20), 0.1),
    ],
)
def test_signals_do_not_look_ahead(close, fn):
    """The value at day t must be identical whether or not later prices exist."""
    full = fn(close)
    for cut in (100, 250, 599):
        partial = fn(close.iloc[: cut + 1])
        pd.testing.assert_series_equal(full.iloc[: cut + 1], partial, check_names=False)


def test_tsmom_values():
    c = pd.Series([1.0, 2.0, 3.0, 2.5, 1.0], index=pd.date_range("2020-01-01", periods=5))
    assert tsmom(c, 1).tolist()[1:] == [1, 1, 0, 0]
    assert tsmom(c, 1, allow_short=True).tolist()[1:] == [1, 1, -1, -1]
    assert np.isnan(tsmom(c, 1).iloc[0])


def test_long_only_bounds(close):
    for fam in FAMILIES.values():
        for p in fam.grid[:3]:
            pos = fam.positions(close, p).dropna()
            assert pos.between(0, 1).all()


def test_vol_target_caps_leverage(close):
    pos = vol_target(buy_and_hold(close), close, target_vol=10.0, window=20, max_leverage=1.0)
    assert pos.max() <= 1.0 + 1e-12


def test_grid_respects_fast_lt_slow():
    assert all(p["fast"] < p["slow"] for p in FAMILIES["ma_cross"].grid)


def test_bad_params():
    c = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        tsmom(c, 0)
    with pytest.raises(ValueError):
        ma_crossover(c, 50, 20)


def test_no_trade_band():
    t = pd.Series([np.nan, 0.50, 0.55, 0.62, 0.0, 0.3, 0.35], index=pd.date_range("2020-01-01", periods=7))
    out = no_trade_band(t, 0.10)
    assert out.tolist()[1:] == [0.50, 0.50, 0.62, 0.0, 0.3, 0.3]
    assert np.isnan(out.iloc[0])
    pd.testing.assert_series_equal(no_trade_band(t, 0.0), t)


def test_band_reduces_turnover(close):
    raw = vol_target(tsmom(close, 30), close, 0.4, 20)
    banded = no_trade_band(raw, 0.1)
    assert banded.diff().abs().sum() < raw.diff().abs().sum()
