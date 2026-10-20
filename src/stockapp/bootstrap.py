from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockapp.kelly import TRADING_DAYS

BLOCK_LENGTH = 20
"""
Days per bootstrap block (about one trading month). Daily returns are not independent: volatility
clusters and there is short-range autocorrelation. Resampling contiguous blocks instead of single days
keeps that structure inside each block, so the bootstrap spread is not understated.
"""

N_RESAMPLES = 1000
"""Number of bootstrap resamples. 1,000 pins the 5th/95th percentiles down to well under a tenth of the band."""

SEED = 42
"""Fixed seed, so the same data always gives the same band (reproducible results and stable caching)."""


@dataclass
class KellyBand:
    p5: float
    p50: float
    p95: float
    prob_positive: float

    @property
    def crosses_zero(self) -> bool:
        """True when the 5-95% band contains 0, i.e. the data can't tell the sign of the edge."""
        return self.p5 <= 0.0 <= self.p95


def block_bootstrap_indices(
    n: int,
    n_resamples: int = N_RESAMPLES,
    block: int = BLOCK_LENGTH,
    seed: int = SEED,
    length: int | None = None,
) -> np.ndarray:
    """
    Moving-block bootstrap indices into a series of n observations, shape (n_resamples, length).

    Each resample concatenates blocks of `block` consecutive indices whose starts are drawn uniformly
    from [0, n - block], truncated to `length` (default n). Fully vectorized. If n < block, the block
    length is reduced to n.
    """
    length = n if length is None else length
    block = min(block, n)
    n_blocks = -(-length // block)  # ceil division
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_resamples, n_blocks * block)
    return idx[:, :length]


def kelly_distribution(e: pd.Series, idx: np.ndarray, prior_sd_annual: float | None = None) -> np.ndarray:
    """
    Gaussian Kelly f* = mean(e) / var(e) for each bootstrap resample (row of idx), computed along axis 1.

    With prior_sd_annual given, each resample's mean is shrunk exactly as in kelly.shrunk_mean:
    mu_post = mean * tau^2 / (tau^2 + var / n) with tau = prior_sd_annual / 252.
    Resamples with variance ~ 0 give 0, matching fractional_kelly.
    """
    samples = e.to_numpy(dtype=float)[idx]
    mean = samples.mean(axis=1)
    var = samples.var(axis=1, ddof=1)
    if prior_sd_annual is not None:
        tau2 = (prior_sd_annual / TRADING_DAYS) ** 2
        mean = mean * tau2 / (tau2 + var / samples.shape[1])
    safe_var = np.where(var > 1e-12, var, 1.0)
    return np.where(var > 1e-12, mean / safe_var, 0.0)


def kelly_band(dist: np.ndarray) -> KellyBand:
    """Summarize a bootstrap Kelly distribution: 5th, 50th and 95th percentiles and P(f* > 0)."""
    p5, p50, p95 = np.percentile(dist, [5, 50, 95])
    return KellyBand(p5=float(p5), p50=float(p50), p95=float(p95), prob_positive=float(np.mean(dist > 0)))
