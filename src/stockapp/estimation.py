from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrunk_means(excess: pd.DataFrame, prior_sd_annual: float, periods_per_year: float) -> pd.Series:
    """
    Per-asset expected excess return per period, shrunk toward 0 with the Tier 0 normal-normal posterior:
    mu_post = mu_hat * tau^2 / (tau^2 + se^2), se^2 = var / n, tau = prior_sd_annual / periods_per_year (the prior
    SD of the per-period mean). Same formula as kelly.shrunk_mean, generalized to weekly data.
    """
    n = excess.count()
    mean, var = excess.mean(), excess.var()
    tau2 = (prior_sd_annual / periods_per_year) ** 2
    se2 = var / n
    return (mean * tau2 / (tau2 + se2)).where(var > 0, mean)


@dataclass(frozen=True)
class CovarianceEstimate:
    sigma: np.ndarray
    """Ledoit-Wolf covariance per period, symmetric positive definite (after any jitter)."""
    shrinkage: float
    """Ledoit-Wolf shrinkage intensity toward the scaled identity, in [0, 1]."""
    condition_sample: float
    condition_shrunk: float
    jitter: float
    """epsilon added to the diagonal to make Cholesky succeed (0 if it already did)."""


def make_spd(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Symmetrize and return (matrix + eps * I, eps) with the smallest eps on a geometric grid (from 1e-12 x mean
    diagonal, x10 per step) for which Cholesky succeeds; eps = 0 if it already does.
    """
    sym = (matrix + matrix.T) / 2
    try:
        np.linalg.cholesky(sym)
        return sym, 0.0
    except np.linalg.LinAlgError:
        pass
    scale = float(np.mean(np.diag(sym))) or 1.0
    identity = np.eye(len(sym))
    for power in range(-12, 3):
        eps = scale * 10.0**power
        try:
            np.linalg.cholesky(sym + eps * identity)
            return sym + eps * identity, eps
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Covariance matrix could not be made positive definite")


def ledoit_wolf(excess: pd.DataFrame, diagnostics: bool = True) -> CovarianceEstimate:
    """
    Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity (sklearn LedoitWolf), with the
    condition number before and after shrinkage (NaN when diagnostics=False, to save time in bootstrap loops),
    followed by a Cholesky check (make_spd).
    """
    values = excess.dropna().to_numpy(dtype=float)
    fit = LedoitWolf().fit(values)
    sigma, jitter = make_spd(np.asarray(fit.covariance_, dtype=float))
    nan = float("nan")
    return CovarianceEstimate(
        sigma=sigma,
        shrinkage=float(fit.shrinkage_),
        condition_sample=float(np.linalg.cond(np.cov(values, rowvar=False, ddof=0))) if diagnostics else nan,
        condition_shrunk=float(np.linalg.cond(sigma)) if diagnostics else nan,
        jitter=jitter,
    )


@dataclass(frozen=True)
class Estimates:
    tickers: list[str]
    mu: np.ndarray
    """Annualized shrunk expected excess returns."""
    mu_sample: np.ndarray
    """Annualized sample mean excess returns (for reference)."""
    sigma: np.ndarray
    """Annualized Ledoit-Wolf covariance."""
    covariance: CovarianceEstimate


def estimate(
    excess: pd.DataFrame, prior_sd_annual: float, periods_per_year: float, diagnostics: bool = True
) -> Estimates:
    """Annualized shrunk means and Ledoit-Wolf covariance (per-period estimates x periods_per_year)."""
    cov = ledoit_wolf(excess, diagnostics)
    return Estimates(
        tickers=[str(c) for c in excess.columns],
        mu=shrunk_means(excess, prior_sd_annual, periods_per_year).to_numpy(dtype=float) * periods_per_year,
        mu_sample=excess.mean().to_numpy(dtype=float) * periods_per_year,
        sigma=cov.sigma * periods_per_year,
        covariance=cov,
    )
