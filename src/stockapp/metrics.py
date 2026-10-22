"""Annualized performance metrics: CAGR, volatility, Sharpe ratio and the approximate growth rate at a fraction."""

import math

import pandas as pd

from stockapp.kelly import TRADING_DAYS, estimate_stats


def cagr(close: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    """
    Compound annual growth rate over the series: (P_end / P_start) ** (P / n) - 1, n = number of daily returns,
    P = trading sessions per year (252 by default; the exchange's own count when known).
    """
    n = len(close) - 1
    if n < 1:
        return 0.0
    return float((close.iloc[-1] / close.iloc[0]) ** (periods_per_year / n) - 1)


def annualized_volatility(r: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    """Annualized volatility: sample stdev of daily returns * sqrt(P), P = sessions per year (default 252)."""
    return estimate_stats(r)[2] * math.sqrt(periods_per_year)


def sharpe_ratio(e: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    """Annualized Sharpe ratio of daily excess returns: mean(e) / stdev(e) * sqrt(P), P default 252. 0 if stdev ~ 0."""
    mu, _, sd = estimate_stats(e)
    return mu / sd * math.sqrt(periods_per_year) if sd > 1e-12 else 0.0


def growth_rate(f: float, rf: pd.Series, e: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    """
    Approximate annualized log growth when holding fraction f in the stock and the rest at rf:
    g(f) ~= P * [mean(rf) + f * mean(e) - f^2 * var(e) / 2], P = sessions per year (default 252).
    This is a second-order (small-return) approximation of E[log(1 + rf + f * e)].
    """
    mu, var, _ = estimate_stats(e)
    return periods_per_year * (float(rf.mean()) + f * mu - f**2 * var / 2)
