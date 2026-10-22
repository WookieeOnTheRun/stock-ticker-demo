import math

import numpy as np
import pandas as pd
import pytest
from fixtures import make_prices

from stockapp.backtest.strategies import BUY_AND_HOLD
from stockapp.backtest.walkforward import walk_forward
from stockapp.signals import SIGNALS
from stockapp.signals.library import (
    bollinger_z,
    donchian_breakout,
    momentum_12_1,
    regime_filter,
    rsi,
    sma_cross,
    vol_target,
)


def test_registry_has_every_signal_with_small_grids():
    assert set(SIGNALS) == {
        "Buy & hold",
        "SMA cross",
        "Donchian breakout",
        "Momentum 12-1",
        "RSI mean reversion",
        "Bollinger z",
        "Vol target",
        "Regime filter",
    }
    assert all(1 <= len(s.grid) <= 4 for s in SIGNALS.values())
    assert "earnings_drift" not in {name.lower().replace(" ", "_") for name in SIGNALS}  # no Tier 4 data


@pytest.mark.parametrize("name", list(SIGNALS))
def test_positions_are_weights_in_unit_interval(name):
    prices = make_prices(800, mu=0.0003, sigma=0.015, seed=3)
    for params in SIGNALS[name].grid:
        position = SIGNALS[name].positions(prices, params)
        assert position.index.equals(prices.index)
        assert position.between(0.0, 1.0).all()


def test_sma_cross_and_momentum_on_a_trend():
    up = pd.Series(np.linspace(100, 200, 400), index=pd.bdate_range("2020-01-01", periods=400))
    assert sma_cross(up, {"fast": 20, "slow": 100}).iloc[-1] == 1.0
    assert sma_cross(up, {"fast": 20, "slow": 100}).iloc[:99].eq(0).all()  # undefined slow SMA -> flat
    assert momentum_12_1(up, {}).iloc[-1] == 1.0
    assert momentum_12_1(up[::-1].set_axis(up.index), {}).iloc[-1] == 0.0


def test_donchian_enters_on_breakout_and_exits_below_half_channel():
    index = pd.bdate_range("2020-01-01", periods=10)
    prices = pd.Series([10, 11, 12, 11, 13, 12, 11, 9, 10, 14.0], index=index)
    pos = donchian_breakout(prices, {"n": 4})
    assert pos.iloc[3] == 0.0  # channel not yet formed
    assert pos.iloc[4] == 1.0  # 13 > prior 4-day high of 12: enter
    assert pos.iloc[5] == 1.0  # 12 is not below the prior 2-day low (11): stay in
    assert pos.iloc[6] == 0.0  # 11 < prior 2-day low min(13, 12) = 12: exit
    assert pos.iloc[8] == 0.0  # 10 does not break the prior 4-day high (13)
    assert pos.iloc[9] == 1.0  # 14 > prior 4-day high max(11, 9, 10, 12) = 12: re-enter


def test_rsi_bounds_and_mean_reversion_entries():
    prices = make_prices(500, mu=0.0, sigma=0.02, seed=9)
    value = rsi(prices).dropna()
    assert value.between(0, 100).all()
    z_pos = bollinger_z(prices, {"n": 20, "z": 2.0})
    z = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()
    assert (z_pos[z < -2.0] == 1.0).all()  # always long right after a 2-SD drop


def test_vol_target_caps_at_one_and_regime_filter_uses_causal_labels():
    prices = make_prices(1200, mu=0.0003, sigma=0.03, seed=4)
    position = vol_target(prices, {"target_ann": 0.10})
    assert position.max() <= 1.0 and position.iloc[-1] < 1.0  # 30%+ vol with a 10% target scales down
    flt = regime_filter(prices, {})
    assert set(flt.unique()) <= {0.0, 1.0}


@pytest.mark.slow
def test_no_signal_beats_buy_and_hold_on_random_walks_on_average():
    """Harness leak check: across 20 random walks, no signal significantly beats buy-and-hold out of sample."""
    diffs: dict[str, list[float]] = {name: [] for name in SIGNALS if name != BUY_AND_HOLD.name}
    for seed in range(20):
        prices = make_prices(2520, mu=0.0, sigma=0.012, seed=500 + seed)
        bh = walk_forward(BUY_AND_HOLD, prices).out_of_sample.metrics["Sharpe"]
        for name in diffs:
            diffs[name].append(walk_forward(SIGNALS[name], prices).out_of_sample.metrics["Sharpe"] - bh)
    for name, values in diffs.items():
        t_stat = np.mean(values) / (np.std(values, ddof=1) / math.sqrt(len(values)))
        assert t_stat < 2, f"{name} shows significant out-of-sample outperformance on random walks (t = {t_stat:.2f})"
