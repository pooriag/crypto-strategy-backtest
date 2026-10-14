"""Performance statistics for daily return series (crypto trades 365 days a year)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PPY = 365


def cagr(r: pd.Series | np.ndarray, ppy: int = PPY) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    if growth <= 0:
        return -1.0
    return float(growth ** (ppy / len(r)) - 1.0)


def ann_vol(r, ppy: int = PPY) -> float:
    return float(np.std(np.asarray(r, dtype=float), ddof=1) * math.sqrt(ppy))


def sharpe(r, ppy: int = PPY, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio of daily returns (rf is an annual rate)."""
    r = np.asarray(r, dtype=float) - rf / ppy
    sd = np.std(r, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean(r) / sd * math.sqrt(ppy))


def sortino(r, ppy: int = PPY) -> float:
    r = np.asarray(r, dtype=float)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if downside == 0:
        return float("nan")
    return float(np.mean(r) / downside * math.sqrt(ppy))


def max_drawdown(r) -> float:
    eq = np.cumprod(1.0 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(np.min(eq / peak - 1.0))


def calmar(r, ppy: int = PPY) -> float:
    mdd = max_drawdown(r)
    return float("nan") if mdd == 0 else cagr(r, ppy) / abs(mdd)


def summary(r: pd.Series, ppy: int = PPY) -> dict:
    """Headline statistics as a flat dict (suitable for a results table)."""
    r = pd.Series(r).dropna()
    return {
        "start": str(r.index[0].date()) if isinstance(r.index, pd.DatetimeIndex) else None,
        "end": str(r.index[-1].date()) if isinstance(r.index, pd.DatetimeIndex) else None,
        "days": len(r),
        "cagr": cagr(r, ppy),
        "ann_vol": ann_vol(r, ppy),
        "sharpe": sharpe(r, ppy),
        "sortino": sortino(r, ppy),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r, ppy),
        "skew": float(pd.Series(r).skew()),
        "kurtosis": float(pd.Series(r).kurt() + 3.0),  # non-excess
    }
