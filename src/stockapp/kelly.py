from dataclasses import dataclass

import pandas as pd


@dataclass
class KellyResult:
    mean_return: float
    variance: float
    stdev: float
    raw_kelly: float
    fraction: float
    applied_kelly: float


def estimate_stats(returns: pd.Series) -> tuple[float, float, float]:
    """Return (mean, sample variance, sample stdev) for a return series."""
    if len(returns) < 2:
        return 0.0, 0.0, 0.0
    return float(returns.mean()), float(returns.var()), float(returns.std())


def fractional_kelly(
    returns: pd.Series, k_fraction: float = 0.5, clamp_min: float = 0.0, clamp_max: float = 1.0
) -> KellyResult:
    """
    Compute fractional Kelly using mean/variance estimate:
    f* = mu / var; then apply fraction: f = k_fraction * f*
    Safety clamps: [clamp_min, clamp_max]. If variance ~ 0, f = 0.
    """
    mu, var, sd = estimate_stats(returns)
    raw = mu / var if var > 1e-12 else 0.0
    frac = max(0.0, min(k_fraction, 1.0))
    applied = max(clamp_min, min(raw * frac, clamp_max))
    return KellyResult(mean_return=mu, variance=var, stdev=sd, raw_kelly=raw, fraction=frac, applied_kelly=applied)
