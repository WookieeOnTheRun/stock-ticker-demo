import math

import numpy as np
import pandas as pd
import pytest
from fixtures import make_prices

from stockapp.backtest.engine import run_backtest
from stockapp.backtest.strategies import BUY_AND_HOLD, Strategy, kelly_buy_and_hold
from stockapp.backtest.walkforward import fixed_params, make_folds, walk_forward


def _random_walk(seed: int, n: int = 5040) -> pd.Series:
    """Zero-drift random walk, about 20 years of business days."""
    return make_prices(n, mu=0.0, sigma=0.01, seed=seed)


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * math.sqrt(252))


def test_engine_shift_prevents_same_bar_foresight():
    prices = _random_walk(seed=1)
    r = prices.pct_change()
    foresight = (r > 0).astype(float)  # perfect foresight if it were applied to the same bar

    result = run_backtest(prices, foresight, cost_bps=0.0)
    unshifted = (foresight * r).dropna()

    assert abs(result.metrics["Sharpe"]) < 0.5
    assert (
        abs(
            result.metrics["Sharpe"]
            - run_backtest(prices, pd.Series(1.0, index=prices.index), cost_bps=0.0).metrics["Sharpe"]
        )
        < 0.5
    )
    assert _sharpe(unshifted) > 10  # what the engine would report without the shift


def test_costs_are_charged_exactly_on_turnover_and_cash_earns_rf():
    index = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.Series([100.0, 101, 102, 103, 104, 105], index=index)
    position = pd.Series([0.0, 1.0, 1.0, 0.25, 0.25, 0.0], index=index)
    rf = pd.Series(0.0002, index=index)

    result = run_backtest(prices, position, rf, cost_bps=5.0, slippage_bps=2.0)

    r = prices.pct_change()
    held = position.shift(1).fillna(0.0)
    turnover = position.diff().abs()
    cost = 7.0 / 1e4
    expected = (held * r + (1 - held) * rf - cost * turnover).iloc[1:]
    np.testing.assert_allclose(result.returns, expected)
    assert result.returns.iloc[0] == pytest.approx(0.0002 - cost)  # day 1: in cash, earns rf, pays the entry
    assert result.turnover.sum() == pytest.approx(2.0)

    cash = run_backtest(prices, pd.Series(0.0, index=index), rf)
    np.testing.assert_allclose(cash.returns, 0.0002)
    assert cash.metrics["Exposure"] == 0


def test_metrics_on_a_known_path():
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.Series([100, 110, 99, 99, 108.9], index=index)
    result = run_backtest(prices, pd.Series([1, 1, 0, 0.5, 0.5], index=index), cost_bps=0.0)
    assert result.metrics["Exposure"] == pytest.approx(0.75)
    assert result.metrics["Avg holding (days)"] == pytest.approx(1.5)  # runs of 2 and 1 days
    assert result.metrics["Hit rate"] == pytest.approx(2 / 3)
    assert result.metrics["Max drawdown"] == pytest.approx(0.1)
    assert result.equity.iloc[0] == 1.0


def _trend(prices: pd.Series, params) -> pd.Series:
    return (prices > prices.rolling(int(params["n"])).mean()).astype(float)


def _reversion(prices: pd.Series, params) -> pd.Series:
    z = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()
    return (z < -float(params["z"])).astype(float)


TOYS = [
    Strategy("trend", "test only", _trend, grid=({"n": 20}, {"n": 50}, {"n": 200})),
    Strategy("reversion", "test only", _reversion, grid=({"z": 1.0}, {"z": 2.0})),
]


@pytest.mark.parametrize("strategy", TOYS, ids=[t.name for t in TOYS])
def test_no_out_of_sample_edge_on_random_walks(strategy):
    diffs = []
    for seed in range(20):
        prices = _random_walk(seed=100 + seed, n=2520)
        oos = walk_forward(strategy, prices).out_of_sample.metrics["Sharpe"]
        bh = walk_forward(BUY_AND_HOLD, prices).out_of_sample.metrics["Sharpe"]
        diffs.append(oos - bh)
    t_stat = np.mean(diffs) / (np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
    assert t_stat < 2  # no significant out-of-sample outperformance on average


def test_folds_expanding_and_rolling():
    index = pd.bdate_range("2010-01-01", "2016-06-30")
    expanding = make_folds(index)
    assert [f[1].year for f in expanding] == [2012, 2013, 2014, 2015]
    assert all(f[0] == index[0] for f in expanding)
    assert expanding[-1][2] == index[-1]  # partial final test year
    rolling = make_folds(index, mode="rolling")
    assert rolling[-1][0] == pd.Timestamp("2013-01-01")
    assert make_folds(pd.bdate_range("2010-01-01", "2012-06-30")) == []


def test_parameters_are_estimated_on_training_data_only_and_switch_causally():
    prices = make_prices(1800, mu=0.0003, sigma=0.01, seed=3)
    seen: list[pd.Timestamp] = []

    def estimate(train_prices, train_rf):
        seen.append(train_prices.index[-1])
        return {"fraction": len(seen) / 10}  # a different constant per fold

    strategy = Strategy(
        "probe", "test only", lambda p, params: pd.Series(params["fraction"], index=p.index), estimate=estimate
    )
    result = walk_forward(strategy, prices)

    assert seen == [fold.train_end for fold in result.folds]
    for i, fold in enumerate(result.folds):
        assert result.positions.loc[fold.train_end] == pytest.approx((i + 1) / 10)  # decided at the training close
    assert result.positions.index.is_unique
    assert result.positions.index.equals(prices.loc[result.folds[0].train_end :].index)
    assert result.trials == 1


def test_grid_search_trials_and_short_history():
    prices = make_prices(1300, mu=0.0003, sigma=0.01, seed=4)
    assert walk_forward(TOYS[0], prices).trials == 3
    with pytest.raises(ValueError, match="more than 3 years"):
        walk_forward(BUY_AND_HOLD, prices.iloc[:500])


def test_kelly_buy_and_hold_fixed_mode_is_in_sample():
    prices = make_prices(1300, mu=0.0008, sigma=0.01, seed=5)
    rf = pd.Series(0.0, index=prices.index)
    strategy = kelly_buy_and_hold("Gaussian", k_fraction=0.5)
    fixed = fixed_params(strategy, prices, {}, rf)
    r = prices.pct_change().dropna()
    expected = min(0.5 * r.mean() / r.var(), 1.0)
    assert fixed.positions.iloc[-1] == pytest.approx(expected)  # estimated on the whole evaluated period
