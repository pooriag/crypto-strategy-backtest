import numpy as np
import pandas as pd
import pytest

from cryptobt.signals import FAMILIES, buy_and_hold, ma_crossover, tsmom, vol_target


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
