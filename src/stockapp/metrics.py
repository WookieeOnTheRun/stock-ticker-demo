import math

import pandas as pd

from stockapp.kelly import TRADING_DAYS, estimate_stats


def cagr(close: pd.Series) -> float:
    """Compound annual growth rate over the series: (P_end / P_start) ** (252 / n) - 1, n = number of daily returns."""
    n = len(close) - 1
    if n < 1:
        return 0.0
    return float((close.iloc[-1] / close.iloc[0]) ** (TRADING_DAYS / n) - 1)


def annualized_volatility(r: pd.Series) -> float:
    """Annualized volatility: sample stdev of daily returns * sqrt(252)."""
    return estimate_stats(r)[2] * math.sqrt(TRADING_DAYS)


def sharpe_ratio(e: pd.Series) -> float:
    """Annualized Sharpe ratio of daily excess returns: mean(e) / stdev(e) * sqrt(252). 0 if stdev ~ 0."""
    mu, _, sd = estimate_stats(e)
    return mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0


def growth_rate(f: float, rf: pd.Series, e: pd.Series) -> float:
    """
    Approximate annualized log growth when holding fraction f in the stock and the rest at rf:
    g(f) ~= 252 * [mean(rf) + f * mean(e) - f^2 * var(e) / 2].
    This is a second-order (small-return) approximation of E[log(1 + rf + f * e)].
    """
    mu, var, _ = estimate_stats(e)
    return TRADING_DAYS * (float(rf.mean()) + f * mu - f**2 * var / 2)
