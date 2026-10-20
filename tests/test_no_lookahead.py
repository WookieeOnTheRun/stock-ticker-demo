"""
Look-ahead guard (Tier 5, A5). For every registered signal (stockapp.backtest.strategies.SIGNALS, every grid point)
and the walk-forward harness itself: compute the output on the full series and on the series truncated at a random
date t; values up to t must be identical. 20 random cuts per function on seeded synthetic data.
Feature functions (Part C) must be added here when they are registered.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from fixtures import make_prices

from stockapp.backtest.strategies import SIGNALS, Strategy, kelly_buy_and_hold
from stockapp.backtest.walkforward import walk_forward

N_CUTS = 20
PRICES = make_prices(1600, mu=0.0003, sigma=0.015, seed=21)  # about 6.3 years of business days
RF = pd.Series(0.0001, index=PRICES.index)


def lookahead_violations(
    fn: Callable[[pd.Series], pd.Series], prices: pd.Series, n_cuts: int = N_CUTS, seed: int = 0, min_t: int = 1
) -> list[pd.Timestamp]:
    """Cut dates at which fn(prices[: t + 1]) differs from fn(prices) on any date up to t."""
    full = fn(prices)
    rng = np.random.default_rng(seed)
    cuts = sorted(rng.choice(np.arange(min_t, len(prices) - 1), size=n_cuts, replace=False))
    bad = []
    for t in cuts:
        cut_date = prices.index[t]
        truncated = fn(prices.iloc[: t + 1])
        expected = full.loc[:cut_date]
        if not truncated.loc[:cut_date].equals(expected):
            bad.append(cut_date)
    return bad


def assert_no_lookahead(fn: Callable[[pd.Series], pd.Series], prices: pd.Series = PRICES, **kwargs) -> None:
    violations = lookahead_violations(fn, prices, **kwargs)
    assert not violations, f"Output up to t changed when data after t was removed, at {violations[:3]}"


SIGNAL_CASES = [
    pytest.param(strategy, params, id=f"{name}-{i}")
    for name, strategy in SIGNALS.items()
    for i, params in enumerate(strategy.grid)
]


@pytest.mark.parametrize(("strategy", "params"), SIGNAL_CASES)
def test_registered_signals_have_no_lookahead(strategy: Strategy, params):
    assert_no_lookahead(lambda p: strategy.positions(p, params))


def _toy_trend(prices: pd.Series, params) -> pd.Series:
    """Test-only causal grid strategy: invested when the close is above its n-day mean."""
    return (prices > prices.rolling(int(params["n"])).mean()).astype(float)


TOY = Strategy("Toy trend", "test only", _toy_trend, grid=({"n": 20}, {"n": 100}))
FIRST_DECISION = int(np.searchsorted(PRICES.index, PRICES.index[0] + pd.DateOffset(years=3))) + 1


@pytest.mark.parametrize("strategy", [kelly_buy_and_hold(), TOY], ids=["kelly-buy-and-hold", "toy-grid"])
def test_walk_forward_positions_have_no_lookahead(strategy):
    def stitched(p: pd.Series) -> pd.Series:
        if p.index[-1] < p.index[0] + pd.DateOffset(years=3):
            return pd.Series(dtype=float)
        return walk_forward(strategy, p, RF.loc[: p.index[-1]]).positions

    assert_no_lookahead(stitched, min_t=FIRST_DECISION)


def test_guard_catches_a_leaky_function():
    def leaky(p: pd.Series) -> pd.Series:
        return (p.shift(-1) > p).astype(float)  # uses tomorrow's close

    def centered(p: pd.Series) -> pd.Series:
        return p.rolling(21, center=True).mean()

    assert lookahead_violations(leaky, PRICES)
    assert lookahead_violations(centered, PRICES)
    assert not lookahead_violations(lambda p: p.rolling(21).mean(), PRICES)
