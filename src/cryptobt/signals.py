"""Trading signals.

Every signal function takes a daily close-price Series and returns the *target
position* decided at the close of day t, using only prices up to and including t.
Positions are in [0, 1] for long-only spot trading (or [-1, 1] with
``allow_short=True``). The backtester is responsible for lagging them, so a
signal must never look at future prices itself - ``tests/test_signals.py``
checks this by truncating the price history.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd


def _direction(x: pd.Series, allow_short: bool) -> pd.Series:
    pos = np.sign(x)
    if not allow_short:
        pos = pos.clip(lower=0)
    return pos.where(x.notna())


def tsmom(close: pd.Series, lookback: int = 60, allow_short: bool = False) -> pd.Series:
    """Time-series momentum: long when the trailing ``lookback``-day return is positive."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    past_return = close / close.shift(lookback) - 1.0
    return _direction(past_return, allow_short).rename("position")


def ma_crossover(close: pd.Series, fast: int = 20, slow: int = 100, allow_short: bool = False) -> pd.Series:
    """Moving-average crossover: long when the fast SMA is above the slow SMA."""
    if not 1 <= fast < slow:
        raise ValueError("need 1 <= fast < slow")
    spread = close.rolling(fast).mean() - close.rolling(slow).mean()
    return _direction(spread, allow_short).rename("position")


def buy_and_hold(close: pd.Series) -> pd.Series:
    """Benchmark: always fully invested once a price exists."""
    return pd.Series(1.0, index=close.index, name="position").where(close.notna())


def vol_target(
    position: pd.Series,
    close: pd.Series,
    target_vol: float = 0.40,
    window: int = 30,
    max_leverage: float = 1.0,
    periods_per_year: int = 365,
) -> pd.Series:
    """Scale a position so its ex-ante annualised volatility is about ``target_vol``.

    Realised volatility over the trailing ``window`` days (known at the close of t)
    is used as the forecast. Exposure is capped at ``max_leverage`` (1.0 = no
    borrowing), so in calm markets the position is not levered up.
    """
    rets = close.pct_change(fill_method=None)
    realised = rets.rolling(window).std() * np.sqrt(periods_per_year)
    scale = (target_vol / realised).clip(upper=max_leverage)
    return (position * scale).rename("position")


def no_trade_band(target: pd.Series, band: float = 0.10) -> pd.Series:
    """Only move to a new target when it differs from the current position by more than ``band``.

    Volatility targeting rescales the position a little every day; most of those
    small trades cost more than they add. A band keeps the position unchanged
    until the gap is material. Moves to or from flat (0) are always executed, so
    entry/exit signals are never delayed. The rule is path-dependent but causal:
    the output at t depends only on targets up to t.
    """
    if band <= 0:
        return target
    vals = target.to_numpy(dtype=float)
    out = np.full_like(vals, np.nan)
    cur = np.nan
    for i, v in enumerate(vals):
        if np.isnan(v):
            cur = np.nan
        elif np.isnan(cur) or v == 0 or cur == 0 or abs(v - cur) > band:
            cur = v
        out[i] = cur
    return pd.Series(out, index=target.index, name="position")


# --------------------------------------------------------------------------- #
# Strategy families used by the walk-forward optimiser
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyFamily:
    name: str
    func: Callable[..., pd.Series]
    grid: tuple[dict, ...]

    def positions(
        self, close: pd.Series, params: dict, vol_targeted: bool = False, band: float = 0.0, **vt_kwargs
    ) -> pd.Series:
        pos = self.func(close, **params)
        if vol_targeted:
            pos = no_trade_band(vol_target(pos, close, **vt_kwargs), band)
        return pos


def _grid(**axes) -> tuple[dict, ...]:
    keys = list(axes)
    combos = [dict(zip(keys, vals)) for vals in product(*axes.values())]
    return tuple(c for c in combos if not ("fast" in c and c["fast"] >= c["slow"]))


FAMILIES: dict[str, StrategyFamily] = {
    "tsmom": StrategyFamily(
        "tsmom", tsmom, _grid(lookback=(7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365))
    ),
    "ma_cross": StrategyFamily(
        "ma_cross", ma_crossover, _grid(fast=(5, 10, 20, 50), slow=(30, 50, 100, 150, 200))
    ),
}
