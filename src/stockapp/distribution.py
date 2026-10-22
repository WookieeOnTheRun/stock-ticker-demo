"""Return-distribution diagnostics: moments, normality and autocorrelation tests, Q-Q data, a Student-t fit and
the Hurst exponent. Used to judge whether Gaussian Kelly's assumptions hold for a stock.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ACF_LAGS = 20
LJUNG_BOX_LAG = 10


def _moments(r: pd.Series) -> tuple[np.ndarray, float]:
    """The non-missing values centered on their mean, and m2, their second central moment (ddof 0)."""
    values = r.dropna().to_numpy(dtype=float)
    centered = values - values.mean()
    return centered, float((centered**2).mean())


def skewness(r: pd.Series) -> float:
    """Moment skewness g1 = m3 / m2^1.5 (m_k = k-th central moment). 0 for a symmetric distribution."""
    centered, m2 = _moments(r)
    return float((centered**3).mean() / m2**1.5) if m2 > 0 else 0.0


def excess_kurtosis(r: pd.Series) -> float:
    """Moment excess kurtosis g2 = m4 / m2^2 - 3. 0 for a normal distribution; > 0 means fatter tails."""
    centered, m2 = _moments(r)
    return float((centered**4).mean() / m2**2 - 3) if m2 > 0 else 0.0


def jarque_bera(r: pd.Series) -> tuple[float, float]:
    """
    Jarque-Bera normality test: JB = n / 6 * (S^2 + K^2 / 4), S = skewness, K = excess kurtosis.
    Under normality JB ~ chi-square with 2 degrees of freedom, whose survival function has the closed form
    exp(-x / 2), so the p-value is exp(-JB / 2) (no scipy needed). Returns (JB, p-value).
    """
    n = int(r.dropna().size)
    jb = n / 6 * (skewness(r) ** 2 + excess_kurtosis(r) ** 2 / 4)
    return float(jb), float(math.exp(-jb / 2))


def _plotting_positions(n: int) -> np.ndarray:
    """Probabilities (i - 0.5) / n for i = 1..n, at which Q-Q plots take the theoretical quantiles."""
    return (np.arange(1, n + 1) - 0.5) / n


def qq_normal(r: pd.Series) -> pd.DataFrame:
    """
    Normal Q-Q data: sorted standardized returns against N(0, 1) quantiles at plotting positions (i - 0.5) / n.
    Points on the 45-degree line mean normal; an S-shape bending away at both ends means fat tails.
    """
    values = np.sort(r.dropna().to_numpy(dtype=float))
    standardized = (values - values.mean()) / values.std(ddof=1)
    theoretical = stats.norm.ppf(_plotting_positions(values.size))
    return pd.DataFrame({"theoretical": theoretical, "sample": standardized})


@dataclass(frozen=True)
class StudentTFit:
    """Maximum-likelihood Student-t parameters: degrees of freedom, location and scale."""

    df: float
    loc: float
    scale: float


def fit_student_t(r: pd.Series) -> StudentTFit:
    """Maximum-likelihood Student-t fit (scipy.stats.t.fit): degrees of freedom, location and scale."""
    df, loc, scale = stats.t.fit(r.dropna().to_numpy(dtype=float))
    return StudentTFit(df=float(df), loc=float(loc), scale=float(scale))


def qq_student_t(r: pd.Series, fit: StudentTFit | None = None) -> pd.DataFrame:
    """Q-Q data against a fitted Student-t: sorted returns vs the fitted t's quantiles at (i - 0.5) / n."""
    fit = fit if fit is not None else fit_student_t(r)
    values = np.sort(r.dropna().to_numpy(dtype=float))
    theoretical = stats.t.ppf(_plotting_positions(values.size), fit.df, loc=fit.loc, scale=fit.scale)
    return pd.DataFrame({"theoretical": theoretical, "sample": values})


def acf(x: pd.Series, nlags: int = ACF_LAGS) -> np.ndarray:
    """Sample autocorrelations at lags 1..nlags: sum((x_t - m)(x_{t-k} - m)) / sum((x_t - m)^2)."""
    values = x.dropna().to_numpy(dtype=float)
    centered = values - values.mean()
    denom = float((centered**2).sum())
    if denom == 0:
        return np.zeros(nlags)
    return np.array([float((centered[k:] * centered[:-k]).sum()) / denom for k in range(1, nlags + 1)])


def acf_band(n: int) -> float:
    """Approximate 95% band for a sample autocorrelation under no autocorrelation: 1.96 / sqrt(n)."""
    return 1.96 / math.sqrt(n)


def ljung_box(x: pd.Series, lag: int = LJUNG_BOX_LAG) -> tuple[float, float]:
    """
    Ljung-Box Q = n (n + 2) * sum_{k=1..lag} rho_k^2 / (n - k); p-value from chi-square with `lag` degrees of
    freedom (scipy). Applied to squared returns it tests for volatility clustering (McLeod-Li). Returns (Q, p).
    """
    n = int(x.dropna().size)
    rho = acf(x, lag)
    q = n * (n + 2) * float(sum(rho[k - 1] ** 2 / (n - k) for k in range(1, lag + 1)))
    return q, float(stats.chi2.sf(q, lag))


def hurst_rs(r: pd.Series, min_window: int = 8) -> float:
    """
    Hurst exponent by rescaled range (R/S): for window sizes from `min_window` to n / 2 (doubling), split the series
    into non-overlapping windows, average R/S (range of cumulative deviations / stdev) per size, and take the slope
    of log(R/S) on log(size). About 0.5 for a random walk, > 0.5 trending, < 0.5 mean-reverting.
    Indicative only; noisy on short samples and biased upward for small windows.
    """
    values = r.dropna().to_numpy(dtype=float)
    sizes, ratios = [], []
    size = min_window
    while size <= values.size // 2:
        chunks = values[: values.size // size * size].reshape(-1, size)
        deviations = np.cumsum(chunks - chunks.mean(axis=1, keepdims=True), axis=1)
        ranges = deviations.max(axis=1) - deviations.min(axis=1)
        sds = chunks.std(axis=1)
        valid = sds > 0
        if valid.any():
            sizes.append(size)
            ratios.append(float((ranges[valid] / sds[valid]).mean()))
        size *= 2
    if len(sizes) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(sizes), np.log(ratios), 1)
    return float(slope)


@dataclass(frozen=True)
class GaussianVerdict:
    """Which Gaussian Kelly assumptions fail, the statistics behind each flag, and a plain message."""

    fat_tails: bool
    non_normal: bool
    volatility_clustering: bool
    excess_kurtosis: float
    jb_p_value: float
    clustering_p_value: float
    message: str


FAT_TAIL_KURTOSIS = 1.0
SIGNIFICANCE = 0.01


def gaussian_kelly_verdict(r: pd.Series) -> GaussianVerdict:
    """
    Plain-language check of the Gaussian Kelly assumptions:
    fat tails if excess kurtosis > 1; non-normal if the Jarque-Bera p-value < 0.01; volatility clustering if the
    Ljung-Box test on squared returns (lag 10) has p < 0.01.
    """
    kurt = excess_kurtosis(r)
    _, jb_p = jarque_bera(r)
    _, lb_p = ljung_box(r**2)
    fat, non_normal, clustering = kurt > FAT_TAIL_KURTOSIS, jb_p < SIGNIFICANCE, lb_p < SIGNIFICANCE
    problems = []
    if fat:
        problems.append(f"fat tails (excess kurtosis {kurt:.1f})")
    if non_normal:
        problems.append("returns are clearly not normal (Jarque-Bera p < 0.01)")
    if clustering:
        problems.append("volatility clusters (calm and turbulent spells)")
    if problems:
        message = (
            "Not entirely: "
            + "; ".join(problems)
            + ". Gaussian Kelly tends to overstate the safe fraction here; prefer the Shrunk or Log-optimal estimate."
        )
    else:
        message = "Reasonably: no strong evidence of fat tails, non-normality or volatility clustering in this sample."
    return GaussianVerdict(fat, non_normal, clustering, kurt, jb_p, lb_p, message)


@dataclass(frozen=True)
class DistributionSummary:
    """Every diagnostic summarize() computes for one return series."""

    verdict: GaussianVerdict
    skewness: float
    jarque_bera: tuple[float, float]
    ljung_box: tuple[float, float]
    ljung_box_squared: tuple[float, float]
    hurst: float
    t_fit: StudentTFit
    qq_normal: pd.DataFrame
    qq_t: pd.DataFrame
    acf: np.ndarray
    acf_squared: np.ndarray
    acf_band: float


def summarize(r: pd.Series) -> DistributionSummary:
    """All distribution diagnostics for one return series."""
    t_fit = fit_student_t(r)
    return DistributionSummary(
        verdict=gaussian_kelly_verdict(r),
        skewness=skewness(r),
        jarque_bera=jarque_bera(r),
        ljung_box=ljung_box(r),
        ljung_box_squared=ljung_box(r**2),
        hurst=hurst_rs(r),
        t_fit=t_fit,
        qq_normal=qq_normal(r),
        qq_t=qq_student_t(r, t_fit),
        acf=acf(r),
        acf_squared=acf(r**2),
        acf_band=acf_band(int(r.dropna().size)),
    )
