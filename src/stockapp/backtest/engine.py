"""Vectorized daily backtest of a target-weight series, with transaction costs and performance metrics."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockapp.kelly import TRADING_DAYS
from stockapp.metrics import cagr
from stockapp.risk import max_drawdown, sortino

DEFAULT_COST_BPS = 5.0  # per side, charged on every change in position


@dataclass(frozen=True)
class BacktestResult:
    """Output of run_backtest: equity curve, daily returns, positions held, turnover and metrics."""

    equity: pd.Series
    """Growth of 1, starting at 1.0 on the first price date."""
    returns: pd.Series
    """Daily strategy returns, net of costs, from the second price date on."""
    positions: pd.Series
    """Weight held during each return day (the target decided at the previous close)."""
    turnover: pd.Series
    """|change in target weight| at each close."""
    metrics: dict[str, float]


def run_backtest(
    prices: pd.Series,
    position: pd.Series,
    rf: pd.Series | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = 0.0,
    periods_per_year: float = TRADING_DAYS,
) -> BacktestResult:
    """
    Daily strategy return = w_{t-1} * r_t + (1 - w_{t-1}) * rf_t - (cost_bps + slippage_bps) / 1e4 * |w_t - w_{t-1}|,
    where w_t in [0, 1] is the target weight decided at the close of t and r_t = P_t / P_{t-1} - 1.
    The one-day shift is what keeps a decision at the close of t from earning the return of day t itself.
    Missing targets (before a signal is defined) mean 0 (cash); the weight before the first date is 0. A trade
    decided at the close of the first date has no return day of its own, so its cost is charged on the second date.
    """
    r = prices.pct_change().iloc[1:]
    target = position.reindex(prices.index).fillna(0.0).clip(0.0, 1.0)
    held = target.shift(1).fillna(0.0).iloc[1:]
    all_turnover = target.diff().fillna(target).abs()
    turnover = all_turnover.iloc[1:].copy()
    if len(turnover):
        turnover.iloc[0] += all_turnover.iloc[0]
    rf_daily = rf.reindex(r.index).fillna(0.0) if rf is not None else pd.Series(0.0, index=r.index)
    cost = (cost_bps + slippage_bps) / 1e4
    returns = (held * r + (1 - held) * rf_daily - cost * turnover).rename("strategy")
    equity = pd.concat([pd.Series([1.0], index=prices.index[:1]), (1 + returns).cumprod()]).rename("equity")
    return BacktestResult(
        equity=equity,
        returns=returns,
        positions=held.rename("position"),
        turnover=turnover.rename("turnover"),
        metrics=compute_metrics(equity, returns, held, turnover, rf_daily, periods_per_year),
    )


def _holding_periods(held: pd.Series) -> list[int]:
    """Lengths (in days) of consecutive runs with a non-zero position."""
    invested = (held > 0).to_numpy()
    lengths, run = [], 0
    for flag in invested:
        if flag:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    held: pd.Series,
    turnover: pd.Series,
    rf: pd.Series,
    periods_per_year: float = TRADING_DAYS,
) -> dict[str, float]:
    """
    CAGR, annualized volatility, Sharpe and Sortino (both on returns in excess of rf), max drawdown (positive
    fraction) and its duration in calendar days, and trading statistics:
      hit rate        share of invested days whose return beat rf
      profit factor   sum of positive excess returns / |sum of negative excess returns| on invested days
      avg holding     mean length in days of consecutive invested runs
      exposure        share of days with a non-zero position
      annual turnover sum of |change in weight| per year (a full entry plus exit counts 2.0)
    """
    excess = returns - rf
    sd = float(returns.std())
    ex_sd = float(excess.std())
    drawdown = max_drawdown(equity)
    invested = held > 0
    inv_excess = excess[invested]
    gains, losses = float(inv_excess[inv_excess > 0].sum()), float(-inv_excess[inv_excess < 0].sum())
    holds = _holding_periods(held)
    years = len(returns) / periods_per_year if len(returns) else float("nan")
    return {
        "CAGR": cagr(equity, periods_per_year),
        "Ann. vol": sd * math.sqrt(periods_per_year),
        "Sharpe": float(excess.mean()) / ex_sd * math.sqrt(periods_per_year) if ex_sd > 1e-12 else 0.0,
        "Sortino": sortino(returns, rf, periods_per_year),
        "Max drawdown": drawdown.depth,
        "Max DD duration (days)": float(drawdown.duration_days),
        "Hit rate": float((inv_excess > 0).mean()) if invested.any() else float("nan"),
        "Profit factor": gains / losses if losses > 0 else (math.inf if gains > 0 else float("nan")),
        "Avg holding (days)": float(np.mean(holds)) if holds else 0.0,
        "Exposure": float(invested.mean()),
        "Annual turnover": float(turnover.sum()) / years if years else float("nan"),
    }
