from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


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


def excess_returns(returns: pd.Series, rf: pd.Series) -> pd.Series:
    """Excess returns e_t = r_t - rf_t, matched by date; dates without an rf value are dropped."""
    return (returns - rf.reindex(returns.index)).dropna()


def fractional_kelly(
    returns: pd.Series,
    k_fraction: float = 0.5,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    rf: pd.Series | None = None,
) -> KellyResult:
    """
    Compute fractional Kelly using mean/variance estimate:
    f* = mu / var; then apply fraction: f = k_fraction * f*
    Safety clamps: [clamp_min, clamp_max]. If variance ~ 0, f = 0.
    With rf given, mu and var are of excess returns e_t = r_t - rf_t, so f* = (mu - r) / sigma^2.
    With rf=None, the raw returns are used, as before.
    """
    if rf is not None:
        returns = excess_returns(returns, rf)
    mu, var, sd = estimate_stats(returns)
    raw = mu / var if var > 1e-12 else 0.0
    frac = max(0.0, min(k_fraction, 1.0))
    applied = max(clamp_min, min(raw * frac, clamp_max))
    return KellyResult(mean_return=mu, variance=var, stdev=sd, raw_kelly=raw, fraction=frac, applied_kelly=applied)


def shrunk_mean(e: pd.Series, prior_sd_annual: float) -> float:
    """
    Posterior mean of the daily expected excess return under a normal-normal model with prior mean 0.

    se^2 = var(e) / n is the squared standard error of the sample mean, tau = prior_sd_annual / 252 is
    the prior SD of the daily mean, and mu_post = mean(e) * tau^2 / (tau^2 + se^2).
    The estimate shrinks toward 0 in proportion to its noise: when se is large relative to tau (short or
    volatile samples) mu_post is pulled close to 0; when se is small the sample mean is barely changed.
    """
    mu, var, _ = estimate_stats(e)
    if var <= 0.0:
        return mu
    se2 = var / len(e)
    tau2 = (prior_sd_annual / TRADING_DAYS) ** 2
    return mu * tau2 / (tau2 + se2)


def shrunk_kelly(e: pd.Series, prior_sd_annual: float) -> float:
    """Kelly fraction from the shrunk mean: f* = mu_post / var(e). Returns 0 if variance ~ 0."""
    _, var, _ = estimate_stats(e)
    return shrunk_mean(e, prior_sd_annual) / var if var > 1e-12 else 0.0


def log_optimal_kelly(r: pd.Series, rf: pd.Series, f_max: float = 10.0, step: float = 0.005) -> float:
    """
    Empirical growth-optimal fraction: argmax over f in [0, f_max] of mean(log1p(R_f)),
    where R_f = rf + f * (r - rf) is the daily portfolio return with fraction f in the stock.

    No Gaussian assumption: the sample's actual tails enter through the log, so the gap between this
    and the mean/variance Kelly is the fat-tail penalty. Any f for which some day has 1 + R_f <= 0
    (total loss) is infeasible and skipped. The grid is evaluated as one (grid x observations) array.
    Returns the unclamped value as a diagnostic; clamp it only when computing an applied fraction.
    """
    rf_aligned = rf.reindex(r.index)
    valid = rf_aligned.notna().to_numpy()
    r_arr = r.to_numpy(dtype=float)[valid]
    rf_arr = rf_aligned.to_numpy(dtype=float)[valid]
    if r_arr.size == 0:
        return 0.0
    grid = np.linspace(0.0, f_max, int(round(f_max / step)) + 1)
    port = rf_arr[None, :] + grid[:, None] * (r_arr - rf_arr)[None, :]
    feasible = (1.0 + port > 0.0).all(axis=1)
    log_growth = np.log1p(port, out=np.full_like(port, -np.inf), where=1.0 + port > 0.0)
    mean_growth = np.where(feasible, log_growth.mean(axis=1), -np.inf)
    return float(grid[int(np.argmax(mean_growth))])
