import numpy as np
import pandas as pd
import pytest

from cryptobt.backtest import Costs
from cryptobt.signals import FAMILIES, StrategyFamily, tsmom
from cryptobt.walkforward import walk_forward


@pytest.fixture
def close():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2015-01-01", periods=6 * 365, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, len(idx)))), index=idx)


def test_oos_period_and_folds(close):
    fam = StrategyFamily("tsmom", tsmom, ({"lookback": 20}, {"lookback": 60}))
    res = walk_forward(close, fam, train_days=730, test_days=180, warmup_days=180)
    folds = res.folds
    # OOS begins after warmup + training and folds tile it without gaps/overlap
    assert pd.Timestamp(folds.test_start.iloc[0]) == close.index[0] + pd.Timedelta(days=180 + 730)
    starts = pd.to_datetime(folds.test_start)
    ends = pd.to_datetime(folds.test_end)
    assert ((starts.iloc[1:].values - ends.iloc[:-1].values) == np.timedelta64(1, "D")).all()
    assert res.backtest.index[0] == starts.iloc[0]
    assert res.backtest.index[-1] == close.index[-1]
    # training windows always end before their test window starts
    assert (pd.to_datetime(folds.train_end) < starts).all()


def test_selection_uses_only_past_data(close):
    """Changing prices *after* a fold's training window must not change its chosen params."""
    fam = FAMILIES["tsmom"]
    base = walk_forward(close, fam, train_days=730, test_days=180, warmup_days=365)
    cut = pd.Timestamp(base.folds.test_start.iloc[2])
    shocked = close.copy()
    shocked.loc[cut:] *= np.exp(np.linspace(0, 3, (shocked.index >= cut).sum()))
    alt = walk_forward(shocked, fam, train_days=730, test_days=180, warmup_days=365)
    assert list(base.folds.params.iloc[:3]) == list(alt.folds.params.iloc[:3])


def test_costs_reduce_returns(close):
    # single grid point so costs cannot change which parameters get selected
    fam = StrategyFamily("tsmom", tsmom, ({"lookback": 14},))
    free = walk_forward(close, fam, costs=Costs(0, 0))
    paid = walk_forward(close, fam, costs=Costs(10, 5))
    assert paid.net.sum() < free.net.sum()
    assert paid.backtest["cost"].sum() > 0


def test_trial_bookkeeping(close):
    fam = FAMILIES["tsmom"]
    res = walk_forward(close, fam)
    assert res.n_trials == len(fam.grid) == len(res.trial_sharpes)
    # no tiny stub fold at the end
    last = res.folds.iloc[-1]
    assert (pd.Timestamp(last.test_end) - pd.Timestamp(last.test_start)).days >= 182 // 4


def test_not_enough_history():
    c = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=100))
    with pytest.raises(ValueError):
        walk_forward(c, FAMILIES["tsmom"])
