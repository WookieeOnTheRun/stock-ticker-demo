"""Strategy definition, the buy-and-hold baselines and the Tier 0 Kelly estimators they size positions with."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from stockapp.drawdown import drawdown_constrained_kelly
from stockapp.kelly import excess_returns, fractional_kelly, log_optimal_kelly, shrunk_kelly

Params = Mapping[str, float | int | str]
PositionFn = Callable[[pd.Series, Params], pd.Series]
"""(prices, params) -> target weight in [0, 1] at each close, using only data up to that close."""
EstimateFn = Callable[[pd.Series, pd.Series], Params]
"""(training prices, training rf) -> parameters estimated from the training window only."""

KELLY_METHODS = ("Gaussian", "Shrunk", "Log-optimal", "Drawdown-constrained")


@dataclass(frozen=True)
class Strategy:
    """
    A position rule with a small, explicit parameter grid. In walk-forward mode, each grid point is tried on the
    training window only and the best is frozen for the test year. `estimate`, if given, adds parameters fitted on
    the training window (e.g. a Kelly fraction) to every grid point.
    """

    name: str
    description: str
    positions: PositionFn
    grid: tuple[Params, ...] = ({},)
    estimate: EstimateFn | None = field(default=None)


def _constant(prices: pd.Series, params: Params) -> pd.Series:
    """Constant target weight params["fraction"] (default 1.0, fully invested) at every close."""
    return pd.Series(float(params.get("fraction", 1.0)), index=prices.index, name="position")


BUY_AND_HOLD = Strategy(
    name="Buy & hold",
    description="Fully invested every day.",
    positions=_constant,
)


def raw_kelly(method: str, r: pd.Series, rf: pd.Series, prior_sd_annual: float = 0.10, max_dd: float = 0.25) -> float:
    """Unclamped Tier 0 Kelly f* for one method on returns r with matching risk-free rates rf."""
    if method == "Gaussian":
        return fractional_kelly(r, rf=rf).raw_kelly
    if method == "Shrunk":
        return shrunk_kelly(excess_returns(r, rf), prior_sd_annual)
    if method == "Log-optimal":
        return log_optimal_kelly(r, rf)
    if method == "Drawdown-constrained":
        return drawdown_constrained_kelly(r, rf, max_dd)
    raise ValueError(f"Unknown Kelly method: {method!r}")


def kelly_buy_and_hold(
    method: str = "Shrunk", k_fraction: float = 0.5, prior_sd_annual: float = 0.10, max_dd: float = 0.25
) -> Strategy:
    """
    Constant-weight buy-and-hold at the Tier 0 applied fraction clip(k_fraction * f*, 0, 1), with f* estimated
    only on the training window passed to `estimate` (never on the period it is then applied to).
    """

    def estimate(train_prices: pd.Series, train_rf: pd.Series) -> Params:
        """Applied fraction from the training window's returns; days without an rf value count as rf = 0."""
        r = train_prices.pct_change().dropna()
        raw = raw_kelly(method, r, train_rf.reindex(r.index).fillna(0.0), prior_sd_annual, max_dd)
        return {"fraction": max(0.0, min(k_fraction * raw, 1.0))}

    return Strategy(
        name="Kelly buy & hold",
        description=f"Constant weight at {k_fraction:g} x {method} Kelly, estimated on the training window.",
        positions=_constant,
        estimate=estimate,
    )
