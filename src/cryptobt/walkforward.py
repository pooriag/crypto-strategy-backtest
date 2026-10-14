"""Walk-forward parameter selection.

For each fold the parameter set with the best *in-sample* Sharpe over the
trailing ``train_days`` is chosen, and then traded - untouched - over the next
``test_days``. Only those out-of-sample days are reported. Folds are stitched
into one target-position series and backtested in a single pass, so switching
costs at fold boundaries are charged like any other trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import Costs, run_backtest
from .metrics import sharpe
from .signals import StrategyFamily


@dataclass
class WalkForwardResult:
    asset: str
    family: str
    vol_targeted: bool
    backtest: pd.DataFrame  # OOS-only daily frame from run_backtest
    folds: pd.DataFrame  # one row per fold with the chosen params
    trial_sharpes: pd.Series = field(repr=False)  # per-period Sharpe of every grid point over the OOS span
    n_trials: int = 0

    @property
    def net(self) -> pd.Series:
        return self.backtest["net"]


def _param_key(p: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in p.items())


def walk_forward(
    close: pd.Series,
    family: StrategyFamily,
    asset: str = "",
    train_days: int = 3 * 365,
    test_days: int = 182,
    warmup_days: int = 365,
    costs: Costs = Costs(),
    vol_targeted: bool = False,
    vt_kwargs: dict | None = None,
    delay: int = 1,
) -> WalkForwardResult:
    close = close.dropna()
    vt_kwargs = vt_kwargs or {}

    # Signals are causal (see tests), so each grid point can be computed once on the
    # full history and sliced later without leaking future information.
    targets = {_param_key(p): family.positions(close, p, vol_targeted, **vt_kwargs) for p in family.grid}
    nets = {k: run_backtest(close, t, costs, delay)["net"] for k, t in targets.items()}
    params = {_param_key(p): p for p in family.grid}

    dates = close.index
    first_train = dates[0] + pd.Timedelta(days=warmup_days)
    test_start = first_train + pd.Timedelta(days=train_days)
    if test_start >= dates[-1]:
        raise ValueError(f"{asset}: not enough history for walk-forward")

    stitched = pd.Series(np.nan, index=dates)
    rows = []
    while test_start <= dates[-1]:
        train_lo = test_start - pd.Timedelta(days=train_days)
        train_hi = test_start - pd.Timedelta(days=1)
        test_hi = min(test_start + pd.Timedelta(days=test_days - 1), dates[-1])
        if (dates[-1] - test_hi).days < test_days // 4:
            test_hi = dates[-1]  # fold a short stub at the end into the last fold

        scores = {k: sharpe(n.loc[train_lo:train_hi]) for k, n in nets.items()}
        best = max(scores, key=lambda k: -np.inf if np.isnan(scores[k]) else scores[k])
        stitched.loc[test_start:test_hi] = targets[best].loc[test_start:test_hi]
        rows.append(
            {
                "train_start": train_lo.date(),
                "train_end": train_hi.date(),
                "test_start": test_start.date(),
                "test_end": test_hi.date(),
                "params": best,
                "train_sharpe": scores[best],
                "test_sharpe": sharpe(nets[best].loc[test_start:test_hi]),
            }
        )
        test_start = test_hi + pd.Timedelta(days=1)

    oos_start = pd.Timestamp(rows[0]["test_start"])
    # Run on the full history so the position entering the first OOS day is charged,
    # then keep OOS days only. Before oos_start the stitched target is NaN => flat.
    bt = run_backtest(close, stitched, costs, delay).loc[oos_start:]
    # The day-before-OOS target is NaN, so the first OOS day is flat; that is the
    # honest outcome (we only start trading once the first model is chosen).
    trial_sharpes = pd.Series(
        {k: _per_period_sharpe(n.loc[oos_start:]) for k, n in nets.items()}, name="per_period_sharpe"
    )
    return WalkForwardResult(
        asset=asset,
        family=family.name,
        vol_targeted=vol_targeted,
        backtest=bt,
        folds=pd.DataFrame(rows),
        trial_sharpes=trial_sharpes,
        n_trials=len(params),
    )


def _per_period_sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")
