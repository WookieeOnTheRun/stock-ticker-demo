"""Drawdown-constrained Kelly: the largest fraction whose bootstrapped 1-year drawdown stays within a limit."""

import numpy as np
import pandas as pd

from stockapp.bootstrap import BLOCK_LENGTH, SEED, block_bootstrap_indices

FRACTION_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)  # 0.00, 0.05, ..., 1.00


def drawdown_constrained_kelly(
    r: pd.Series,
    rf: pd.Series,
    max_dd: float,
    confidence: float = 0.90,
    horizon: int = 252,
    n_paths: int = 1000,
    block: int = BLOCK_LENGTH,
    seed: int = SEED,
) -> float:
    """
    Largest fraction f in {0, 0.05, ..., 1} whose `confidence` quantile of max drawdown over `horizon`
    days is <= max_dd; 0 if none qualifies.

    Paths are block-bootstrap resamples of the observed (r, rf) pairs, of length `horizon`. The same
    paths are used for every f (common random numbers), so results are stable across f. Each f is one
    vectorized (paths x days) computation; looping over the 21 fractions instead of building a single
    (f x paths x days) array keeps peak memory near 10 MB rather than about 200 MB. For each f the equity
    curve is cumprod(1 + rf + f * (r - rf)), and the max drawdown is max(1 - equity / running peak), with
    the running peak starting at 1.
    """
    rf_aligned = rf.reindex(r.index)
    valid = rf_aligned.notna().to_numpy()
    r_arr = r.to_numpy(dtype=float)[valid]
    rf_arr = rf_aligned.to_numpy(dtype=float)[valid]
    if r_arr.size < 2:
        return 0.0
    idx = block_bootstrap_indices(r_arr.size, n_resamples=n_paths, block=block, seed=seed, length=horizon)
    rf_paths = rf_arr[idx]
    excess_paths = r_arr[idx] - rf_paths
    dd_quantile = np.empty(FRACTION_GRID.size)
    for i, f in enumerate(FRACTION_GRID):
        equity = np.cumprod(1.0 + (rf_paths + f * excess_paths), axis=1)
        peak = np.maximum(np.maximum.accumulate(equity, axis=1), 1.0)
        dd_quantile[i] = np.quantile((1.0 - equity / peak).max(axis=1), confidence)
    qualifying = FRACTION_GRID[dd_quantile <= max_dd]
    return float(qualifying.max()) if qualifying.size else 0.0
