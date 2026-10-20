"""
Deterministic action rules (BUY / ADD, SELL / TRIM, HOLD) that combine the calibrated model probability, Tier 0
shrunk Kelly with its bootstrap band, the causal volatility regime and drawdown-constrained Kelly. Every clause is
reported, and the rule itself is backtested walk-forward so its out-of-sample record is shown next to it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockapp.backtest.strategies import Params, Strategy
from stockapp.bootstrap import block_bootstrap_indices, kelly_band, kelly_distribution
from stockapp.drawdown import drawdown_constrained_kelly
from stockapp.kelly import excess_returns, shrunk_kelly
from stockapp.model.probability import ModelReport
from stockapp.regimes import vol_regimes

P_BUY = 0.55
P_SELL = 0.45
TRIM_TOLERANCE = 0.05
NO_SKILL_P = 0.5
MIN_SCORED_FOR_SKILL = 252  # out-of-sample predictions with known outcomes before skill can be established
BANNER = "Educational tool. Not investment advice. Past performance does not predict future results."


def advice_probability(report: ModelReport) -> float:
    """The probability the rules act on: the model's latest calibrated p if it has out-of-sample skill, else 0.5."""
    return report.latest_probability if report.has_skill else NO_SKILL_P


@dataclass(frozen=True)
class AdviceInputs:
    p: float
    """Calibrated probability, or 0.5 when the model has no demonstrated skill."""
    kelly_p5: float
    """5th percentile of the bootstrap shrunk-Kelly distribution (raw f*)."""
    target: float
    """Applied fractional Kelly: clip(k * shrunk f*, 0, 1)."""
    regime: str | None
    dd_kelly: float
    """Drawdown-constrained Kelly fraction (0..1)."""
    current: float = 0.0


@dataclass(frozen=True)
class Advice:
    action: str
    new_position: float
    clauses: dict[str, bool]
    inputs: AdviceInputs


def decide(inputs: AdviceInputs, p_buy: float = P_BUY, p_sell: float = P_SELL) -> Advice:
    """
    SELL / TRIM if the current position exceeds target + 0.05, or p <= p_sell, or the regime is high-volatility and
    drawdown-constrained Kelly is below the current position. Checked first, so an over-target position is trimmed
    even when the buy clauses hold. New position = the minimum over the clauses that fired: target, 0, dd_kelly.
    BUY / ADD if the shrunk-Kelly 5th percentile > 0 and p >= p_buy and the regime is not high-volatility; the new
    position is the applied fractional Kelly target. HOLD otherwise (position unchanged).
    """
    over_target = inputs.current > inputs.target + TRIM_TOLERANCE
    low_p = inputs.p <= p_sell
    high_vol_cut = inputs.regime == "high" and inputs.dd_kelly < inputs.current
    edge = inputs.kelly_p5 > 0
    high_p = inputs.p >= p_buy
    calm = inputs.regime != "high"
    clauses = {
        "SELL: current > target + 0.05": over_target,
        f"SELL: p <= {p_sell:g}": low_p,
        "SELL: high-vol regime and drawdown-constrained Kelly < current": high_vol_cut,
        "BUY: shrunk Kelly 5th percentile > 0": edge,
        f"BUY: p >= {p_buy:g}": high_p,
        "BUY: regime is not high-vol": calm,
    }
    if over_target or low_p or high_vol_cut:
        caps = [inputs.target] * over_target + [0.0] * low_p + [inputs.dd_kelly] * high_vol_cut
        return Advice("SELL / TRIM", min(inputs.current, *caps), clauses, inputs)
    if edge and high_p and calm:
        return Advice("BUY / ADD", inputs.target, clauses, inputs)
    return Advice("HOLD", inputs.current, clauses, inputs)


@dataclass(frozen=True)
class KellyInputs:
    target: float
    kelly_p5: float
    dd_kelly: float


def kelly_inputs(
    prices: pd.Series, rf: pd.Series, k_fraction: float, prior_sd_annual: float, max_dd: float
) -> KellyInputs:
    """Applied shrunk-Kelly target, the bootstrap 5th percentile of shrunk f*, and drawdown-constrained Kelly."""
    r = prices.pct_change().dropna()
    rf_aligned = rf.reindex(r.index).fillna(0.0)
    e = excess_returns(r, rf_aligned)
    band = kelly_band(kelly_distribution(e, block_bootstrap_indices(len(e)), prior_sd_annual))
    return KellyInputs(
        target=max(0.0, min(k_fraction * shrunk_kelly(e, prior_sd_annual), 1.0)),
        kelly_p5=band.p5,
        dd_kelly=drawdown_constrained_kelly(r, rf_aligned, max_dd),
    )


def causal_skill_gate(probabilities: pd.Series, labels: pd.Series, climatology: pd.Series, horizon: int) -> pd.Series:
    """
    The probability to act on at each date: the model's p if, using only predictions whose outcomes were already
    known on that date (label end <= date), its Brier score beats climatology over at least MIN_SCORED_FOR_SKILL of
    them; otherwise 0.5. This keeps the "no skill -> 0.5" rule free of look-ahead in backtests.
    """
    p = probabilities.to_numpy(dtype=float)
    y = labels.reindex(probabilities.index).to_numpy(dtype=float)
    c = climatology.reindex(probabilities.index).to_numpy(dtype=float)
    err_model = np.where(np.isnan(y), 0.0, (p - y) ** 2)
    err_clim = np.where(np.isnan(y), 0.0, (c - y) ** 2)
    scored = (~np.isnan(y)).astype(float)
    # outcome of the prediction made at i is known at i + horizon, so shift the running sums by horizon
    cum_model = pd.Series(np.cumsum(err_model)).shift(horizon).fillna(0.0).to_numpy()
    cum_clim = pd.Series(np.cumsum(err_clim)).shift(horizon).fillna(0.0).to_numpy()
    cum_n = pd.Series(np.cumsum(scored)).shift(horizon).fillna(0.0).to_numpy()
    skilled = (cum_n >= MIN_SCORED_FOR_SKILL) & (cum_model < cum_clim)
    return pd.Series(np.where(skilled, p, NO_SKILL_P), index=probabilities.index, name="p_used")


def rule_strategy(
    p_used: pd.Series,
    k_fraction: float,
    prior_sd_annual: float,
    max_dd: float,
    p_buy: float = P_BUY,
    p_sell: float = P_SELL,
) -> Strategy:
    """
    The advice rule as a walk-forward strategy. Kelly inputs are estimated on each training window and frozen for
    the test year; p (already causally gated) and the causal regime are read daily; the position carries over on
    HOLD. Dates before p is available use 0.5.
    """

    def estimate(train_prices: pd.Series, train_rf: pd.Series) -> Params:
        k = kelly_inputs(train_prices, train_rf, k_fraction, prior_sd_annual, max_dd)
        return {"target": k.target, "kelly_p5": k.kelly_p5, "dd_kelly": k.dd_kelly}

    def positions(prices: pd.Series, params: Params) -> pd.Series:
        regimes = vol_regimes(prices.pct_change())
        p = p_used.reindex(prices.index).fillna(NO_SKILL_P).to_numpy()
        labels = regimes.to_numpy()
        out = np.zeros(len(prices))
        current = 0.0
        for i in range(len(prices)):
            regime = labels[i] if isinstance(labels[i], str) else None
            advice = decide(
                AdviceInputs(
                    p=float(p[i]),
                    kelly_p5=float(params.get("kelly_p5", 0.0)),
                    target=float(params.get("target", 0.0)),
                    regime=regime,
                    dd_kelly=float(params.get("dd_kelly", 0.0)),
                    current=current,
                ),
                p_buy,
                p_sell,
            )
            current = advice.new_position
            out[i] = current
        return pd.Series(out, index=prices.index, name="position")

    return Strategy("Advice rule", "The BUY / SELL / HOLD rule traded walk-forward.", positions, estimate=estimate)


FAN_PERCENTILES = (5, 25, 50, 75, 95)


def fan_chart_paths(
    prices: pd.Series, rf: pd.Series, fraction: float, horizon: int = 252, n_paths: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """
    1-year Monte Carlo fan of equity at a constant fraction using the Tier 0 block-bootstrap (20-day blocks) of the
    observed (r, rf) pairs: equity = cumprod(1 + rf + f (r - rf)). Columns are the 5/25/50/75/95th percentiles
    for days 0..horizon. Resampled history, not a forecast.
    """
    r = prices.pct_change().dropna()
    rf_aligned = rf.reindex(r.index).fillna(0.0).to_numpy()
    idx = block_bootstrap_indices(len(r), n_resamples=n_paths, seed=seed, length=horizon)
    r_paths, rf_paths = r.to_numpy()[idx], rf_aligned[idx]
    equity = np.cumprod(1 + rf_paths + fraction * (r_paths - rf_paths), axis=1)
    equity = np.hstack([np.ones((n_paths, 1)), equity])
    return pd.DataFrame({f"p{q}": np.percentile(equity, q, axis=0) for q in FAN_PERCENTILES})
