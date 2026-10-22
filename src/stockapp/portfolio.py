"""Portfolio page analytics: estimation, every weighting method, bootstrap bands, frontier and a split-sample test."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockapp.estimation import Estimates, estimate
from stockapp.optimize import (
    Allocation,
    ConstrainedKelly,
    HRPResult,
    efficient_frontier,
    hrp_weights,
    unconstrained_kelly,
)
from stockapp.portfolio_bootstrap import WeightBands, bootstrap_weights
from stockapp.portfolio_data import ReturnsData
from stockapp.portfolio_eval import fixed_weight_growth, split_halves

EQUAL = "Equal weight"
HRP = "HRP"
UNCONSTRAINED = "Fractional Kelly, unconstrained (reference)"
CONSTRAINED = "Constrained Kelly"


@dataclass(frozen=True)
class PortfolioAnalysis:
    """Everything analyze_portfolio computes, ready to display."""

    estimates: Estimates
    weights: pd.DataFrame
    """Rows = tickers, columns = weighting methods. The Constrained Kelly column is NaN if the optimizer failed."""
    allocation: Allocation
    """The constrained Kelly result: weights, or the reason the optimizer failed."""
    hrp: HRPResult
    bands: WeightBands
    frontier: pd.DataFrame
    marks: dict[str, tuple[float, float]]
    """(annualized volatility, expected excess return) of each long-only weighting that has weights."""
    split_growth: pd.DataFrame
    """Second-half growth of $1 with weights estimated on the first half."""
    split_date: pd.Timestamp
    split_failure: str | None
    """Why Constrained Kelly is missing from split_growth (the first-half solve failed); None if it is there."""


def _weights_table(
    returns: pd.DataFrame, est: Estimates, solver: ConstrainedKelly, c: float
) -> tuple[pd.DataFrame, Allocation, HRPResult]:
    """
    Weights per method (equal, HRP, unconstrained and constrained Kelly) for one set of estimates. If the constrained
    optimizer fails, its column is NaN: callers must leave it out rather than read it as weights.
    """
    tickers = est.tickers
    allocation = solver.solve(est.mu, est.sigma, c)
    hrp = hrp_weights(returns)
    weights = pd.DataFrame(
        {
            EQUAL: np.full(len(tickers), 1 / len(tickers)),
            HRP: hrp.weights.reindex(tickers).to_numpy(),
            UNCONSTRAINED: c * unconstrained_kelly(est.mu, est.sigma),
            CONSTRAINED: allocation.weights if allocation.weights is not None else np.full(len(tickers), np.nan),
        },
        index=tickers,
    )
    return weights, allocation, hrp


def analyze_portfolio(
    data: ReturnsData,
    prior_sd_annual: float,
    c: float,
    w_max: float = 0.25,
    gross_max: float = 1.0,
    sector_caps: dict[str, float] | None = None,
    sectors: list[str | None] | None = None,
) -> PortfolioAnalysis:
    """
    Estimate mu (shrunk) and Sigma (Ledoit-Wolf), compute equal-weight, HRP, unconstrained fractional Kelly
    (c * Sigma^-1 mu, reference only) and constrained Kelly weights, bootstrap bands for the constrained weights,
    the long-only frontier, and a growth-of-$1 comparison with weights estimated on the first half of the data and
    evaluated on the second half. A failed constrained solve leaves Constrained Kelly out of the marks and the
    split-sample curves, and failed bootstrap resamples are left out of the bands; nothing substitutes for them.
    """
    n = data.excess.shape[1]
    solver = ConstrainedKelly(
        n, long_only=True, gross_max=gross_max, w_max=w_max, sector_caps=sector_caps, sectors=sectors
    )
    est = estimate(data.excess, prior_sd_annual, data.periods_per_year)
    weights, allocation, hrp = _weights_table(data.returns, est, solver, c)
    bands = bootstrap_weights(data.excess, prior_sd_annual, data.periods_per_year, solver, c)
    frontier = efficient_frontier(est.mu, est.sigma, solver)
    marks = {}
    for name in (EQUAL, HRP, CONSTRAINED):
        if name == CONSTRAINED and allocation.failed:
            continue
        w = weights[name].to_numpy()
        marks[name] = (float(np.sqrt(w @ est.sigma @ w)), float(w @ est.mu))

    first_ret, second_ret = split_halves(data.returns)
    first_ex = data.excess.loc[first_ret.index]
    first_est = estimate(first_ex, prior_sd_annual, data.periods_per_year, diagnostics=False)
    first_weights, first_allocation, _ = _weights_table(first_ret, first_est, solver, c)
    if first_allocation.failed:
        # fixed_weight_growth would read the NaN weights as 0, i.e. all cash: drop the curve instead.
        first_weights = first_weights.drop(columns=CONSTRAINED)
    split_growth = fixed_weight_growth(
        second_ret, data.rf, {name: first_weights[name] for name in first_weights.columns}
    )
    return PortfolioAnalysis(
        estimates=est,
        weights=weights,
        allocation=allocation,
        hrp=hrp,
        bands=bands,
        frontier=frontier,
        marks=marks,
        split_growth=split_growth,
        split_date=pd.Timestamp(second_ret.index[0]),
        split_failure=first_allocation.reason,
    )
