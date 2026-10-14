import numpy as np
import pandas as pd
import pytest

from cryptobt.backtest import Costs, equal_weight_portfolio, run_backtest
from cryptobt.metrics import cagr, max_drawdown, sharpe, summary


def _series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="D"), dtype=float)


def test_position_is_lagged_one_day():
    close = _series([100, 110, 121, 133.1])
    target = _series([1, 1, 0, 0])
    bt = run_backtest(close, target, Costs(0, 0))
    # target decided at close of day 0 earns day 1's return, etc.
    assert bt["held"].tolist() == [0, 1, 1, 0]
    assert bt["net"].round(10).tolist() == [0, 0.1, 0.1, 0]


def test_engine_does_not_add_lookahead_of_its_own():
    """Sanity check of the timing: only a signal that peeks at the future can 'win'
    on a random walk, and an extra day of delay destroys that fake edge.

    (The engine executes whatever target it is given, so look-ahead has to be kept
    out of the signals themselves - that is tested in test_signals.py.)
    """
    rng = np.random.default_rng(1)
    close = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
    peeking = (close.shift(-1) > close).astype(float)  # uses tomorrow's close
    cheat = run_backtest(close, peeking, Costs(0, 0))
    assert sharpe(cheat["net"]) > 5  # absurd Sharpe = the classic look-ahead symptom
    lagged = run_backtest(close, peeking, Costs(0, 0), delay=2)
    assert abs(sharpe(lagged["net"])) < 2


def test_costs_charged_on_turnover():
    close = _series([100, 100, 100, 100, 100])
    target = _series([1, 0, 1, 1, 0])
    bt = run_backtest(close, target, Costs(fee_bps=10, slippage_bps=5))
    # held = [0,1,0,1,1]; turnover = [0,1,1,1,0]
    assert bt["turnover"].tolist() == [0, 1, 1, 1, 0]
    assert bt["cost"].sum() == pytest.approx(3 * 0.0015)
    assert bt["equity"].iloc[-1] == pytest.approx((1 - 0.0015) ** 3)


def test_delay_must_be_positive():
    with pytest.raises(ValueError):
        run_backtest(_series([1, 2]), _series([1, 1]), delay=0)


def test_buy_and_hold_matches_asset():
    close = _series([100, 105, 95, 120])
    bt = run_backtest(close, _series([1, 1, 1, 1]), Costs(0, 0))
    # first day is spent entering, afterwards returns match the asset
    assert bt["equity"].iloc[-1] == pytest.approx(120 / 100)


def test_equal_weight_handles_late_start():
    a = _series([0.01, 0.01, 0.01])
    b = _series([np.nan, 0.03, 0.03])
    port = equal_weight_portfolio(pd.DataFrame({"a": a, "b": b}))
    assert port.round(10).tolist() == [0.01, 0.02, 0.02]


def test_metrics_basic():
    r = _series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert cagr(_series([0.0] * 365)) == 0
    assert cagr(_series([0.01] * 365)) == pytest.approx(1.01**365 - 1)
    s = summary(r)
    assert s["days"] == 3 and s["start"] == "2020-01-01"


def test_sharpe_known_value():
    r = np.array([0.01, -0.01] * 50) + 0.001
    expected = 0.001 / np.std(r, ddof=1) * np.sqrt(365)
    assert sharpe(r) == pytest.approx(expected)
