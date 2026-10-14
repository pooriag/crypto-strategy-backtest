import numpy as np
import pandas as pd
import pytest

from cryptobt.stats import (
    bootstrap_ci,
    deflated_sharpe,
    expected_max_sharpe,
    paired_bootstrap_test,
    probabilistic_sharpe,
    stationary_bootstrap_indices,
)


def _r(mu, sd=0.02, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.date_range("2018-01-01", periods=n))


def test_indices_shape_and_range():
    idx = stationary_bootstrap_indices(100, 50, mean_block=10, seed=1)
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_mean_block_length_is_respected():
    idx = stationary_bootstrap_indices(5000, 20, mean_block=25, seed=2)
    breaks = (np.diff(idx, axis=1) != 1) & (np.diff(idx, axis=1) != -4999)
    avg_block = idx.size / (breaks.sum() + idx.shape[0])
    assert 20 < avg_block < 30


def test_ci_contains_estimate():
    ci = bootstrap_ci(_r(0.001), n_boot=300)
    for _, row in ci.iterrows():
        assert row.ci_low <= row.estimate <= row.ci_high


def test_paired_test_detects_clear_edge_and_no_edge():
    bench = _r(0.0, seed=3)
    better = bench + 0.002  # same noise, strictly better every day
    res = paired_bootstrap_test(better, bench, n_boot=300)
    assert res["observed_diff"] > 0 and res["p_value"] < 0.01
    same = paired_bootstrap_test(bench, bench, n_boot=100)
    assert same["observed_diff"] == 0


def test_psr_behaviour():
    assert probabilistic_sharpe(_r(0.003)) > 0.99
    assert probabilistic_sharpe(_r(-0.003)) < 0.01
    noise = probabilistic_sharpe(_r(0.0, seed=5))
    assert 0.01 < noise < 0.99


def test_more_trials_raise_the_bar():
    rng = np.random.default_rng(0)
    trials = rng.normal(0, 0.02, 50)
    assert expected_max_sharpe(trials, 50) > expected_max_sharpe(trials, 5) > 0
    r = _r(0.0015)
    assert deflated_sharpe(r, trials, 50) < probabilistic_sharpe(r)


def test_bad_block():
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(10, 2, 0.5)
