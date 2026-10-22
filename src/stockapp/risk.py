"""Single-asset risk analytics: VaR / CVaR, drawdowns, volatility estimates, beta and risk-adjusted ratios."""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from stockapp.kelly import TRADING_DAYS
from stockapp.metrics import cagr


def historical_var(r: pd.Series, alpha: float = 0.05) -> float:
    """Historical value-at-risk: the loss not exceeded with probability 1 - alpha, -quantile(r, alpha), positive."""
    return float(-np.quantile(r.to_numpy(dtype=float), alpha))


def historical_cvar(r: pd.Series, alpha: float = 0.05) -> float:
    """Historical conditional VaR (expected shortfall): -mean(r | r <= quantile(r, alpha)), as a positive number."""
    values = r.to_numpy(dtype=float)
    cutoff = np.quantile(values, alpha)
    return float(-values[values <= cutoff].mean())


def parametric_var(r: pd.Series, alpha: float = 0.05) -> float:
    """Normal (variance-covariance) VaR: -(mean + stdev * Phi^-1(alpha)), as a positive number."""
    return float(-(r.mean() + r.std() * NormalDist().inv_cdf(alpha)))


@dataclass(frozen=True)
class Drawdown:
    """The largest drawdown of a series, with its dates and duration."""

    depth: float
    """Largest peak-to-trough fall, as a positive fraction (0.35 = -35%)."""
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    """First date the price regained the peak, or None if it hasn't."""
    duration_days: int
    """Calendar days from peak to recovery, or to the last date if not recovered; 0 if there was no drawdown."""


def max_drawdown(prices: pd.Series) -> Drawdown:
    """Maximum drawdown of a price series: depth = max over t of 1 - P_t / max(P_0..P_t)."""
    values = prices.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(prices.index)
    drawdowns = 1 - values / np.maximum.accumulate(values)
    trough = int(np.argmax(drawdowns))
    peak = int(np.argmax(values[: trough + 1]))
    depth = float(drawdowns[trough])
    regained = np.flatnonzero(values[trough:] >= values[peak]) + trough
    recovery = dates[regained[0]] if depth > 0 and regained.size else None
    end = recovery if recovery is not None else dates[-1]
    return Drawdown(
        depth=depth,
        peak_date=dates[peak],
        trough_date=dates[trough],
        recovery_date=recovery,
        duration_days=int((end - dates[peak]).days) if depth > 0 else 0,
    )


def rolling_vol(r: pd.Series, window: int = 21, periods_per_year: float = TRADING_DAYS) -> pd.Series:
    """Annualized rolling volatility: stdev of the last `window` daily returns * sqrt(periods_per_year)."""
    return r.rolling(window).std() * math.sqrt(periods_per_year)


def ewma_vol(r: pd.Series, lam: float = 0.94, periods_per_year: float = TRADING_DAYS) -> pd.Series:
    """
    Annualized EWMA volatility, RiskMetrics style: sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r_t^2 (zero mean),
    times periods_per_year, square-rooted. lam = 0.94 is the RiskMetrics daily convention. The value at t uses
    returns up to and including t, so it is the forecast for t + 1.
    """
    variance: pd.Series = (r**2).ewm(alpha=1 - lam, adjust=False).mean()
    return (variance * periods_per_year) ** 0.5


@dataclass(frozen=True)
class BetaFit:
    """OLS fit of daily returns on benchmark returns."""

    beta: float
    alpha_annual: float
    """Intercept (daily) * periods_per_year."""
    r2: float
    n_obs: int


def beta_r2(r: pd.Series, bench_r: pd.Series, periods_per_year: float = TRADING_DAYS) -> BetaFit:
    """OLS of r on benchmark returns over their common dates: r_t = a + beta * b_t + e_t."""
    joined = pd.concat([r, bench_r], axis=1, join="inner").dropna()
    y = joined.iloc[:, 0].to_numpy(dtype=float)
    x = joined.iloc[:, 1].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ np.array([a, b])
    total = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid**2).sum() / total if total > 0 else 0.0
    return BetaFit(beta=float(b), alpha_annual=float(a * periods_per_year), r2=float(r2), n_obs=len(y))


def sortino(r: pd.Series, rf: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    """
    Annualized Sortino ratio: mean(e) / sqrt(mean(min(e, 0)^2)) * sqrt(periods_per_year), with e = r - rf
    (downside deviation measured against 0 excess return). 0 if there are no losing days.
    """
    e = (r - rf.reindex(r.index)).dropna()
    downside = math.sqrt(float((np.minimum(e.to_numpy(dtype=float), 0.0) ** 2).mean())) if len(e) else 0.0
    return float(e.mean()) / downside * math.sqrt(periods_per_year) if downside > 1e-12 else 0.0


def calmar(prices: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    """Calmar ratio: CAGR / |max drawdown| over the series. 0 if there was no drawdown."""
    depth = max_drawdown(prices).depth
    return cagr(prices, periods_per_year) / depth if depth > 1e-12 else 0.0


def omega(r: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: sum(max(r - threshold, 0)) / sum(max(threshold - r, 0)). inf if there are no losses."""
    values = r.to_numpy(dtype=float)
    gains = np.maximum(values - threshold, 0.0).sum()
    losses = np.maximum(threshold - values, 0.0).sum()
    return float(gains / losses) if losses > 0 else math.inf


@dataclass(frozen=True)
class GarchForecast:
    """GARCH(1,1) forecast: horizon-average and daily volatility, fitted parameters and whether the fit converged."""

    annualized_vol: float
    """Average forecast volatility over the horizon, annualized."""
    path: np.ndarray
    """Annualized forecast volatility for each day 1..horizon."""
    params: dict[str, float]
    converged: bool


def garch_vol_forecast(r: pd.Series, horizon: int = 21, periods_per_year: float = TRADING_DAYS) -> GarchForecast:
    """
    GARCH(1,1) volatility forecast with the arch package (arch 8.0): returns are scaled to percent for numerical
    stability, a constant-mean GARCH(1,1) with normal errors is fitted, and the analytic variance forecast for
    days 1..horizon is converted back to annualized volatility.
    """
    from arch import arch_model  # imported lazily: only this function needs arch

    model = arch_model(r.to_numpy(dtype=float) * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    result = model.fit(disp="off", show_warning=False)
    variance = result.forecast(horizon=horizon, reindex=False).variance.to_numpy()[-1]
    path = np.sqrt(variance * periods_per_year) / 100
    return GarchForecast(
        annualized_vol=float(math.sqrt(variance.mean() * periods_per_year) / 100),
        path=path,
        params={str(k): float(v) for k, v in result.params.items()},
        converged=result.convergence_flag == 0,
    )


@dataclass(frozen=True)
class RiskSummary:
    """Every non-GARCH risk analytic from summarize()."""

    historical_var: float
    historical_cvar: float
    parametric_var: float
    drawdown: Drawdown
    sortino: float
    calmar: float
    omega: float
    rolling_vol: pd.Series
    ewma_vol: pd.Series


def summarize(r: pd.Series, prices: pd.Series, rf: pd.Series, periods_per_year: float = TRADING_DAYS) -> RiskSummary:
    """All non-GARCH risk analytics for one return series (95% VaR/CVaR, threshold-0 Omega)."""
    return RiskSummary(
        historical_var=historical_var(r),
        historical_cvar=historical_cvar(r),
        parametric_var=parametric_var(r),
        drawdown=max_drawdown(prices),
        sortino=sortino(r, rf, periods_per_year),
        calmar=calmar(prices, periods_per_year),
        omega=omega(r),
        rolling_vol=rolling_vol(r, periods_per_year=periods_per_year),
        ewma_vol=ewma_vol(r, periods_per_year=periods_per_year),
    )
