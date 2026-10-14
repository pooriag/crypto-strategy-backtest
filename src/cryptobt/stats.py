"""Statistical checks: is a backtest result distinguishable from luck?

* **Stationary block bootstrap** (Politis & Romano, 1994) - resamples blocks of
  random geometric length so volatility clustering and autocorrelation in daily
  crypto returns are preserved. Used for confidence intervals on CAGR / Sharpe /
  drawdown and for a *paired* test of strategy vs. buy-and-hold (both series are
  resampled with the same indices, so their correlation is kept).
* **Probabilistic and Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2012/2014) -
  the probability that the true Sharpe exceeds a benchmark, corrected for
  non-normal returns, and, for the DSR, for the number of strategy variants that
  were tried before picking the best one.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, seed: int | None = 0) -> np.ndarray:
    """Return an (n_boot, n) array of resampled time indices (circular blocks)."""
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    rng = np.random.default_rng(seed)
    p_new = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new_block = rng.random((n_boot, n)) < p_new
    fresh = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


DEFAULT_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "cagr": metrics.cagr,
    "sharpe": metrics.sharpe,
    "max_drawdown": metrics.max_drawdown,
}


def bootstrap_ci(
    returns: pd.Series,
    stats: dict[str, Callable] | None = None,
    n_boot: int = 2000,
    mean_block: float = 20.0,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Point estimate and (1 - alpha) percentile interval for each statistic."""
    stats = stats or DEFAULT_STATS
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    idx = stationary_bootstrap_indices(len(r), n_boot, mean_block, seed)
    samples = r[idx]
    rows = {}
    for name, fn in stats.items():
        dist = np.array([fn(s) for s in samples])
        rows[name] = {
            "estimate": fn(r),
            "ci_low": float(np.nanquantile(dist, alpha / 2)),
            "ci_high": float(np.nanquantile(dist, 1 - alpha / 2)),
        }
    return pd.DataFrame(rows).T


def paired_bootstrap_test(
    strategy: pd.Series,
    benchmark: pd.Series,
    stat: Callable[[np.ndarray], float] = metrics.sharpe,
    n_boot: int = 2000,
    mean_block: float = 20.0,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> dict:
    """Bootstrap the difference ``stat(strategy) - stat(benchmark)`` on aligned days.

    Returns the observed difference, its percentile CI and a one-sided bootstrap
    p-value: the share of resamples in which the strategy did *not* beat the
    benchmark. Small values mean the advantage is robust to resampling.
    """
    df = pd.concat({"s": strategy, "b": benchmark}, axis=1).dropna()
    s, b = df["s"].to_numpy(), df["b"].to_numpy()
    idx = stationary_bootstrap_indices(len(df), n_boot, mean_block, seed)
    diffs = np.array([stat(s[i]) - stat(b[i]) for i in idx])
    return {
        "observed_diff": stat(s) - stat(b),
        "ci_low": float(np.nanquantile(diffs, alpha / 2)),
        "ci_high": float(np.nanquantile(diffs, 1 - alpha / 2)),
        "p_value": float(np.mean(diffs <= 0)),
        "n_days": len(df),
    }


# --------------------------------------------------------------------------- #
# Probabilistic / Deflated Sharpe ratio
# --------------------------------------------------------------------------- #


def _per_period_sharpe(r: np.ndarray) -> float:
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd) if sd > 0 else float("nan")


def probabilistic_sharpe(returns, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``) given skew/kurtosis of returns.

    ``sr_benchmark`` is per period (daily), not annualised.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(r)
    sr = _per_period_sharpe(r)
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt()) + 3.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if n < 3 or not np.isfinite(sr) or denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return _N.cdf(z)


def expected_max_sharpe(trial_sharpes, n_trials: int | None = None) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe ratios under the null of no skill."""
    srs = np.asarray(trial_sharpes, dtype=float)
    srs = srs[np.isfinite(srs)]
    n = n_trials or len(srs)
    if n < 2:
        return 0.0
    sd = float(np.std(srs, ddof=1))
    return sd * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * math.e))
    )


def deflated_sharpe(returns, trial_sharpes, n_trials: int | None = None) -> float:
    """Deflated Sharpe ratio: PSR against the Sharpe you'd expect from the best of N lucky trials."""
    return probabilistic_sharpe(returns, expected_max_sharpe(trial_sharpes, n_trials))
