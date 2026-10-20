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
    paths are used for every f (common random numbers), so results are stable across f and the search
    is one vectorized (f x paths x days) computation. For each f the equity curve is
    cumprod(1 + rf + f * (r - rf)), and the max drawdown is max(1 - equity / running peak), with the
    running peak starting at 1.
    """
    rf_aligned = rf.reindex(r.index)
    valid = rf_aligned.notna().to_numpy()
    r_arr = r.to_numpy(dtype=float)[valid]
    rf_arr = rf_aligned.to_numpy(dtype=float)[valid]
    if r_arr.size < 2:
        return 0.0
    idx = block_bootstrap_indices(r_arr.size, n_resamples=n_paths, block=block, seed=seed, length=horizon)
    r_paths, rf_paths = r_arr[idx], rf_arr[idx]
    port = rf_paths[None, :, :] + FRACTION_GRID[:, None, None] * (r_paths - rf_paths)[None, :, :]
    equity = np.cumprod(1.0 + port, axis=2)
    peak = np.maximum(np.maximum.accumulate(equity, axis=2), 1.0)
    max_drawdown = (1.0 - equity / peak).max(axis=2)
    dd_quantile = np.quantile(max_drawdown, confidence, axis=1)
    qualifying = FRACTION_GRID[dd_quantile <= max_dd]
    return float(qualifying.max()) if qualifying.size else 0.0
