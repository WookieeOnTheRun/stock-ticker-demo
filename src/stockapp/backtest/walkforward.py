from dataclasses import dataclass
from typing import Literal

import pandas as pd

from stockapp.backtest.engine import DEFAULT_COST_BPS, BacktestResult, run_backtest
from stockapp.backtest.strategies import Params, Strategy
from stockapp.kelly import TRADING_DAYS

TRAIN_YEARS = 3
TEST_YEARS = 1
SELECTION_METRIC = "Sharpe"
TRIALS_CAPTION = "More trials make it likelier that the best result is luck."


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    """Last training date; parameters are chosen at this close and first applied to the next day's return."""
    test_end: pd.Timestamp
    """Last date whose return this fold's parameters earn."""
    params: Params
    train_metrics: dict[str, float]


@dataclass(frozen=True)
class WalkForwardResult:
    strategy: str
    out_of_sample: BacktestResult
    """Stitched test-year segments: every day's position was decided with parameters chosen on earlier data."""
    in_sample: BacktestResult
    """The final fold's chosen parameters run over the training window they were chosen on."""
    folds: list[Fold]
    positions: pd.Series
    """Stitched out-of-sample target weights, from the first fold's last training date on."""
    trials: int
    """Parameter combinations evaluated (the size of the strategy's grid)."""


def make_folds(
    index: pd.DatetimeIndex,
    train_years: int = TRAIN_YEARS,
    test_years: int = TEST_YEARS,
    mode: Literal["expanding", "rolling"] = "expanding",
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    (train_start, train_end, test_end) date triples. Test year k starts train_years + k * test_years after the first
    date; training is everything before it (expanding) or the preceding train_years (rolling). The last test segment
    may be shorter than test_years. Returns [] if there is not enough data for one fold.
    """
    first, last = index[0], index[-1]
    folds = []
    k = 0
    while True:
        test_start = first + pd.DateOffset(years=train_years + k * test_years)
        if test_start > last:
            break
        train_from = first if mode == "expanding" else test_start - pd.DateOffset(years=train_years)
        train = index[(index >= train_from) & (index < test_start)]
        test = index[(index >= test_start) & (index < test_start + pd.DateOffset(years=test_years))]
        if len(train) and len(test):
            folds.append((train[0], train[-1], test[-1]))
        k += 1
    return folds


def _choose(
    strategy: Strategy,
    train_prices: pd.Series,
    train_rf: pd.Series,
    cost_bps: float,
    slippage_bps: float,
    periods_per_year: float,
) -> tuple[Params, dict[str, float]]:
    """Best grid point by training-window Sharpe (first wins ties), with any training-fitted estimates merged in."""
    estimated = strategy.estimate(train_prices, train_rf) if strategy.estimate else {}
    best: tuple[Params, dict[str, float]] | None = None
    for grid_point in strategy.grid:
        params = {**grid_point, **estimated}
        result = run_backtest(
            train_prices,
            strategy.positions(train_prices, params),
            train_rf,
            cost_bps,
            slippage_bps,
            periods_per_year,
        )
        if best is None or result.metrics[SELECTION_METRIC] > best[1][SELECTION_METRIC]:
            best = (params, result.metrics)
    if best is None:
        raise ValueError(f"Strategy {strategy.name!r} declares an empty parameter grid")
    return best


def walk_forward(
    strategy: Strategy,
    prices: pd.Series,
    rf: pd.Series | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = 0.0,
    train_years: int = TRAIN_YEARS,
    test_years: int = TEST_YEARS,
    mode: Literal["expanding", "rolling"] = "expanding",
    periods_per_year: float = TRADING_DAYS,
) -> WalkForwardResult:
    """
    Walk-forward evaluation: for each fold, choose parameters on the training window only, then freeze them for
    the following test year. Fold k's parameters set the target weight at every close from its last training date
    up to (not including) the next fold's last training date, so each decision uses only data available at that
    close. The stitched positions are backtested once, so costs at fold boundaries are charged too.
    """
    rf_full = rf.reindex(prices.index).fillna(0.0) if rf is not None else pd.Series(0.0, index=prices.index)
    folds_dates = make_folds(pd.DatetimeIndex(prices.index), train_years, test_years, mode)
    if not folds_dates:
        raise ValueError(f"Walk-forward needs more than {train_years} years of data")
    folds: list[Fold] = []
    segments: list[pd.Series] = []
    last_train_prices = prices.iloc[:0]
    for i, (train_start, train_end, test_end) in enumerate(folds_dates):
        train_prices = prices.loc[train_start:train_end]
        params, train_metrics = _choose(
            strategy, train_prices, rf_full.loc[train_start:train_end], cost_bps, slippage_bps, periods_per_year
        )
        folds.append(Fold(train_start, train_end, test_end, params, train_metrics))
        decisions = strategy.positions(prices.loc[:test_end], params)
        segment = decisions.loc[train_end:test_end]
        if i < len(folds_dates) - 1:
            segment = segment.iloc[:-1]  # the next fold decides at this fold's last test close
        segments.append(segment)
        last_train_prices = train_prices
    positions = pd.concat(segments).rename("position")
    oos_prices = prices.loc[positions.index[0] :]
    out_of_sample = run_backtest(oos_prices, positions, rf_full, cost_bps, slippage_bps, periods_per_year)
    final = folds[-1]
    in_sample = run_backtest(
        last_train_prices,
        strategy.positions(last_train_prices, final.params),
        rf_full,
        cost_bps,
        slippage_bps,
        periods_per_year,
    )
    return WalkForwardResult(strategy.name, out_of_sample, in_sample, folds, positions, len(strategy.grid))


def fixed_params(
    strategy: Strategy,
    prices: pd.Series,
    params: Params,
    rf: pd.Series | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = 0.0,
    periods_per_year: float = TRADING_DAYS,
) -> BacktestResult:
    """
    Run one parameter set over the whole period. This is in-sample by construction: if the strategy estimates
    anything (e.g. a Kelly fraction), it is estimated on the same period it is evaluated on.
    """
    rf_full = rf.reindex(prices.index).fillna(0.0) if rf is not None else pd.Series(0.0, index=prices.index)
    estimated = strategy.estimate(prices, rf_full) if strategy.estimate else {}
    return run_backtest(
        prices, strategy.positions(prices, {**params, **estimated}), rf_full, cost_bps, slippage_bps, periods_per_year
    )
