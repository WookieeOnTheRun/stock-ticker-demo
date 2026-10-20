"""
Model features and labels. Every feature at t uses only prices (and benchmark prices) up to t; each is registered
in FEATURES so the look-ahead guard checks it. Labels look forward by construction and are never features.
"""

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from stockapp.regimes import vol_regimes
from stockapp.risk import ewma_vol
from stockapp.signals.library import rsi

TRADING_DAYS = 252
HORIZONS = (5, 21)
WARMUP = 252  # rows before every feature (incl. 252-day ones) is defined are dropped

FeatureFn = Callable[[pd.Series, pd.Series | None], pd.Series]


def _ret(n: int) -> FeatureFn:
    return lambda p, b: p / p.shift(n) - 1


def _sma_distance(n: int) -> FeatureFn:
    return lambda p, b: p / p.rolling(n).mean() - 1


def _realized_vol(p: pd.Series, b: pd.Series | None) -> pd.Series:
    return p.pct_change().rolling(21).std() * math.sqrt(TRADING_DAYS)


def _ewma_vol_ratio(p: pd.Series, b: pd.Series | None) -> pd.Series:
    """EWMA volatility over 252-day realized volatility: > 1 when recent volatility is above its norm."""
    r = p.pct_change()
    return ewma_vol(r) / (r.rolling(TRADING_DAYS).std() * math.sqrt(TRADING_DAYS))


def _regime(p: pd.Series, b: pd.Series | None) -> pd.Series:
    """Causal Tier 2 volatility regime as 0 (low), 1 (normal), 2 (high); NaN before labels exist."""
    return vol_regimes(p.pct_change()).map({"low": 0.0, "normal": 1.0, "high": 2.0}).astype(float)


def _drawdown_252(p: pd.Series, b: pd.Series | None) -> pd.Series:
    return p / p.rolling(TRADING_DAYS, min_periods=1).max() - 1


def _beta_252(p: pd.Series, b: pd.Series | None) -> pd.Series:
    if b is None:
        return pd.Series(np.nan, index=p.index)
    r, rb = p.pct_change(), b.reindex(p.index).pct_change()
    return r.rolling(TRADING_DAYS).cov(rb) / rb.rolling(TRADING_DAYS).var()


def _relative_strength_63(p: pd.Series, b: pd.Series | None) -> pd.Series:
    if b is None:
        return pd.Series(np.nan, index=p.index)
    bench = b.reindex(p.index)
    return (p / p.shift(63)) - (bench / bench.shift(63))


FEATURES: dict[str, FeatureFn] = {
    "ret_5": _ret(5),
    "ret_21": _ret(21),
    "ret_63": _ret(63),
    "ret_126": _ret(126),
    "ret_252": _ret(252),
    "dist_sma50": _sma_distance(50),
    "dist_sma200": _sma_distance(200),
    "rsi_14": lambda p, b: rsi(p, 14),
    "vol_21": _realized_vol,
    "ewma_vol_ratio": _ewma_vol_ratio,
    "regime": _regime,
    "drawdown_252": _drawdown_252,
    "beta_252": _beta_252,
    "rel_strength_63": _relative_strength_63,
}
"""Registered features. The IV-minus-realized-vol spread is omitted: Tier 4 implied volatility is not available."""

BENCHMARK_FEATURES = {"beta_252", "rel_strength_63"}


def build_features(prices: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """All registered features on the price dates; benchmark features are dropped if no benchmark is given."""
    names = [n for n in FEATURES if benchmark is not None or n not in BENCHMARK_FEATURES]
    return pd.DataFrame({n: FEATURES[n](prices, benchmark) for n in names}, index=prices.index)


def forward_excess_label(prices: pd.Series, rf_daily: pd.Series, horizon: int) -> pd.Series:
    """
    1.0 if the return from close t to close t + horizon beats the risk-free rate compounded over the same days,
    else 0.0; NaN for the last `horizon` dates, whose outcome is not known yet. Uses future data by design.
    """
    forward = prices.shift(-horizon) / prices - 1
    rf = rf_daily.reindex(prices.index).fillna(0.0)
    rf_forward: pd.Series = np.exp(np.log1p(rf).rolling(horizon).sum().shift(-horizon)) - 1
    label: pd.Series = (forward - rf_forward > 0).astype(float)
    return label.where(forward.notna())
