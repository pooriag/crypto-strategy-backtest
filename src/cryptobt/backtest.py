"""Vectorised daily backtester with explicit timing and trading costs.

Timing convention
-----------------
* ``target[t]`` is decided from information available at the close of day t.
* It is traded at the next available close and held over the following day, so the
  position earning day t's return is ``held[t] = target[t - delay]`` (``delay=1``
  by default; ``delay=2`` adds a full day of execution latency as a stress test).
* Trading costs are charged on the day the position changes:
  ``cost[t] = |held[t] - held[t-1]| * (fee_bps + slippage_bps) / 10_000``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    fee_bps: float = 10.0  # typical spot taker fee (0.10%)
    slippage_bps: float = 5.0  # half-spread + market impact for small size

    @property
    def per_unit_turnover(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000.0


def run_backtest(
    close: pd.Series,
    target: pd.Series,
    costs: Costs = Costs(),
    delay: int = 1,
) -> pd.DataFrame:
    """Simulate one asset. Returns a daily frame with positions, returns and equity.

    Columns: ``asset_ret, held, turnover, gross, cost, net, equity``.
    Days before the first valid signal are flat (held = 0).
    """
    if delay < 1:
        raise ValueError("delay must be >= 1 (a position cannot use the same close it was decided on)")
    close, target = close.align(target, join="left")
    asset_ret = close.pct_change(fill_method=None).fillna(0.0)
    held = target.shift(delay).fillna(0.0)
    turnover = held.diff().abs()
    turnover.iloc[0] = abs(held.iloc[0])
    gross = held * asset_ret
    cost = turnover * costs.per_unit_turnover
    net = gross - cost
    out = pd.DataFrame(
        {
            "asset_ret": asset_ret,
            "held": held,
            "turnover": turnover,
            "gross": gross,
            "cost": cost,
            "net": net,
        }
    )
    out["equity"] = (1.0 + out["net"]).cumprod()
    return out


def equal_weight_portfolio(net_returns: pd.DataFrame) -> pd.Series:
    """Equal-weight daily-rebalanced mix of per-asset strategy returns.

    On each day the weight is split across the assets that have data that day,
    so a late-starting asset (ETH) joins the portfolio when its history begins.
    Rebalancing costs between sleeves are ignored (they are second-order relative
    to the signal turnover already charged inside each sleeve).
    """
    return net_returns.mean(axis=1, skipna=True).fillna(0.0).rename("net")


def to_equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def time_in_market(held: pd.Series) -> float:
    return float((held.abs() > 1e-12).mean())


def annual_turnover(turnover: pd.Series, periods_per_year: int = 365) -> float:
    return float(np.nanmean(turnover) * periods_per_year)
