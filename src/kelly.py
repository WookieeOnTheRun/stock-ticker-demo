from dataclasses import dataclass
import numpy as np
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
    """Return (mean, variance, stdev) for a return series."""
    if returns is None or len(returns) < 2:
        return 0.0, 0.0, 0.0
    mu = float(np.mean(returns))
    var = float(np.var(returns, ddof=1))  # sample variance
    sd = float(np.sqrt(var))
    return mu, var, sd


def fractional_kelly(returns: pd.Series, k_fraction: float = 0.5, clamp_min: float = 0.0, clamp_max: float = 1.0) -> KellyResult:
    """
    Compute fractional Kelly using mean/variance estimate:
    f* = mu / var; then apply fraction: f = k_fraction * f*
    Safety clamps: [clamp_min, clamp_max]. If variance ~ 0, f = 0.
    """
    mu, var, sd = estimate_stats(returns)
    if var <= 1e-12:
        raw = 0.0
    else:
        raw = mu / var
    raw = float(raw)
    frac = float(max(0.0, min(k_fraction, 1.0)))
    applied = raw * frac
    applied = max(clamp_min, min(applied, clamp_max))
    return KellyResult(mean_return=mu, variance=var, stdev=sd, raw_kelly=raw, fraction=frac, applied_kelly=applied)
