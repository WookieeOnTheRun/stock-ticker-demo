"""
Causal signal library. Each signal maps (prices, params) to a target weight in [0, 1] at every close, using only
data up to that close (enforced by tests/test_no_lookahead.py), with a small explicit parameter grid.
"""

import numpy as np
import pandas as pd

from stockapp.backtest.strategies import Params, Strategy
from stockapp.regimes import vol_regimes
from stockapp.risk import ewma_vol


def _hold_between(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """1 from an entry close until the next exit close (exclusive of the exit), 0 otherwise; exits win ties."""
    state = pd.Series(np.nan, index=enter.index)
    state[enter.fillna(False).astype(bool)] = 1.0
    state[exit_.fillna(False).astype(bool)] = 0.0
    return state.ffill().fillna(0.0)


def sma_cross(prices: pd.Series, params: Params) -> pd.Series:
    """1 while the fast simple moving average is above the slow one, else 0 (0 until both exist)."""
    fast = prices.rolling(int(params["fast"])).mean()
    slow = prices.rolling(int(params["slow"])).mean()
    return (fast > slow).astype(float)


def donchian_breakout(prices: pd.Series, params: Params) -> pd.Series:
    """
    Enter on a close above the prior n-day high, exit on a close below the prior n/2-day low. Channels use closes
    and are shifted one day, so today's close is compared with the channel formed before it.
    """
    n = int(params["n"])
    upper = prices.rolling(n).max().shift(1)
    lower = prices.rolling(n // 2).min().shift(1)
    return _hold_between(prices > upper, prices < lower)


def momentum_12_1(prices: pd.Series, params: Params) -> pd.Series:
    """1 if the return from 252 to 21 trading days ago (12 months, skipping the latest month) is positive."""
    momentum = prices.shift(21) / prices.shift(252) - 1
    return (momentum > 0).astype(float)


def rsi(prices: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI: 100 - 100 / (1 + avg gain / avg loss), with exponential smoothing alpha = 1/n."""
    change = prices.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-change).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def rsi_mean_reversion(prices: pd.Series, params: Params) -> pd.Series:
    """Buy when RSI(n) falls below `lo` (oversold); sell when it rises above `hi` (overbought)."""
    value = rsi(prices, int(params["n"]))
    return _hold_between(value < float(params["lo"]), value > float(params["hi"]))


def bollinger_z(prices: pd.Series, params: Params) -> pd.Series:
    """Mean reversion: buy when the close is more than z rolling SDs below its n-day mean; sell back at the mean."""
    n = int(params["n"])
    z = (prices - prices.rolling(n).mean()) / prices.rolling(n).std()
    return _hold_between(z < -float(params["z"]), z > 0)


def vol_target(prices: pd.Series, params: Params) -> pd.Series:
    """
    Volatility-targeting overlay on buy-and-hold: weight = min(1, target / EWMA volatility), where the EWMA
    (lambda 0.94) volatility at t uses returns up to t. Use it to scale any position as a multiplier.
    """
    vol = ewma_vol(prices.pct_change())
    return (float(params["target_ann"]) / vol).clip(upper=1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def regime_filter(prices: pd.Series, params: Params) -> pd.Series:
    """
    0 in the high-volatility regime, 1 otherwise, using only the causal Tier 2 labels (21-day volatility against
    expanding-window percentiles). Never uses full-sample HMM states. Invested (1) before labels exist.
    """
    labels = vol_regimes(prices.pct_change())
    return pd.Series(np.where(labels == "high", 0.0, 1.0), index=prices.index)


SIGNAL_STRATEGIES = [
    Strategy(
        "SMA cross",
        "Long while the fast SMA is above the slow SMA.",
        sma_cross,
        grid=tuple({"fast": f, "slow": s} for f in (20, 50) for s in (100, 200)),
    ),
    Strategy(
        "Donchian breakout",
        "Enter above the prior n-day high, exit below the prior n/2-day low.",
        donchian_breakout,
        grid=({"n": 55}, {"n": 100}),
    ),
    Strategy("Momentum 12-1", "Long if the 12-month return excluding the last month is positive.", momentum_12_1),
    Strategy(
        "RSI mean reversion",
        "Buy when RSI(14) is oversold, sell when overbought.",
        rsi_mean_reversion,
        grid=tuple({"n": 14, "lo": lo, "hi": hi} for lo in (25, 30) for hi in (70, 75)),
    ),
    Strategy(
        "Bollinger z",
        "Buy z SDs below the 20-day mean, sell back at the mean.",
        bollinger_z,
        grid=({"n": 20, "z": 1.5}, {"n": 20, "z": 2.0}),
    ),
    Strategy(
        "Vol target",
        "Buy-and-hold scaled to a target volatility (EWMA), capped at 100%.",
        vol_target,
        grid=({"target_ann": 0.10}, {"target_ann": 0.15}),
    ),
    Strategy("Regime filter", "Out of the market in the causal high-volatility regime.", regime_filter),
]
