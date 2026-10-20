import time

import numpy as np
import pandas as pd
import pytest
from fixtures import make_prices

from stockapp.bootstrap import block_bootstrap_indices, kelly_band, kelly_distribution
from stockapp.drawdown import drawdown_constrained_kelly


def _returns(n=1250, seed=3):
    r = make_prices(n + 1, mu=0.0004, sigma=0.02, seed=seed).pct_change().dropna()
    return r, pd.Series(0.00016, index=r.index)


def test_bootstrap_same_seed_is_identical():
    np.testing.assert_array_equal(block_bootstrap_indices(500, seed=7), block_bootstrap_indices(500, seed=7))


def test_bootstrap_shape_and_range():
    n = 1250
    idx = block_bootstrap_indices(n)
    assert idx.shape == (1000, n)
    assert idx.min() >= 0
    assert idx.max() <= n - 1


def test_kelly_band_percentiles_are_ordered():
    r, rf = _returns()
    band = kelly_band(kelly_distribution(r - rf, block_bootstrap_indices(len(r))))
    assert band.p5 <= band.p50 <= band.p95
    assert 0.0 <= band.prob_positive <= 1.0


def test_drawdown_constrained_is_monotone_and_bounded():
    r, rf = _returns()
    limits = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    results = [drawdown_constrained_kelly(r, rf, max_dd) for max_dd in limits]
    assert all(0.0 <= f <= 1.0 for f in results)
    assert all(tighter <= looser for looser, tighter in zip(results, results[1:], strict=False))


@pytest.mark.slow
def test_performance_budgets():
    r, rf = _returns(n=1250)
    e = r - rf
    drawdown_constrained_kelly(r, rf, 0.25)  # warm-up

    start = time.perf_counter()
    kelly_distribution(e, block_bootstrap_indices(len(e), n_resamples=1000))
    assert time.perf_counter() - start < 0.5

    start = time.perf_counter()
    drawdown_constrained_kelly(r, rf, 0.25)
    assert time.perf_counter() - start < 1.5
