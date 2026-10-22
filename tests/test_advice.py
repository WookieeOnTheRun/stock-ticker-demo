from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixtures import make_prices

from stockapp.advice import (
    MIN_SCORED_FOR_SKILL,
    AdviceInputs,
    advice_probability,
    causal_skill_gate,
    decide,
    fan_chart_paths,
    rule_strategy,
    skill_established,
)
from stockapp.backtest.walkforward import walk_forward


def _inputs(**overrides):
    base = {"p": 0.6, "kelly_p5": 0.1, "target": 0.4, "regime": "normal", "dd_kelly": 0.8, "current": 0.0}
    return AdviceInputs(**(base | overrides))


def test_buy_when_edge_probability_and_calm_regime():
    advice = decide(_inputs())
    assert advice.action == "BUY / ADD" and advice.new_position == 0.4
    assert advice.clauses["BUY: shrunk Kelly 5th percentile > 0"]


def test_hold_without_edge_or_conviction():
    assert decide(_inputs(kelly_p5=-0.2)).action == "HOLD"
    assert decide(_inputs(p=0.5)).action == "HOLD"
    held = decide(_inputs(p=0.5, current=0.3))
    assert held.action == "HOLD" and held.new_position == 0.3


def _report(n_known, informative=True, horizon=21, seed=2):
    """
    A stand-in ModelReport shaped like walk_forward_model's: out-of-sample probabilities on every decision date, the
    last `horizon` outcomes not yet known, climatology 0.6 and has_skill computed the same way (Brier vs climatology).
    """
    index = pd.bdate_range("2020-01-01", periods=n_known + horizon)
    outcomes = (np.random.default_rng(seed).random(len(index)) < 0.6).astype(float)
    probabilities = pd.Series(outcomes * 0.8 + 0.1 if informative else 0.9 - outcomes * 0.8, index=index)
    labels = pd.Series(outcomes, index=index)
    labels.iloc[-horizon:] = np.nan
    climatology = pd.Series(0.6, index=index)
    known = labels.notna()
    brier = float(((probabilities - labels)[known] ** 2).mean())
    brier_climatology = float(((climatology - labels)[known] ** 2).mean())
    return SimpleNamespace(
        horizon=horizon,
        probabilities=probabilities,
        labels=labels,
        climatology=climatology,
        latest_probability=0.9,
        has_skill=brier < brier_climatology,
        metrics={"Out-of-sample predictions": float(known.sum())},
    )


def test_no_skill_path_uses_half_and_cannot_buy():
    no_skill = _report(900, informative=False)
    assert not no_skill.has_skill and not skill_established(no_skill)
    assert advice_probability(no_skill) == 0.5
    assert decide(_inputs(p=advice_probability(no_skill))).action == "HOLD"
    skilled = _report(900)
    assert skill_established(skilled) and advice_probability(skilled) == 0.9


def test_beating_climatology_needs_enough_scored_predictions():
    too_few = _report(MIN_SCORED_FOR_SKILL - 1)
    assert too_few.has_skill  # beats climatology...
    assert not skill_established(too_few) and advice_probability(too_few) == 0.5  # ...but on too few predictions
    just_enough = _report(MIN_SCORED_FOR_SKILL)
    assert skill_established(just_enough) and advice_probability(just_enough) == 0.9


@pytest.mark.parametrize("n_known", [100, MIN_SCORED_FOR_SKILL - 1, MIN_SCORED_FOR_SKILL, 600])
@pytest.mark.parametrize("informative", [True, False])
def test_live_gate_is_the_track_record_gate_on_the_latest_date(n_known, informative):
    report = _report(n_known, informative)
    gate = causal_skill_gate(report.probabilities, report.labels, report.climatology, report.horizon)
    track_record_trusts_latest = gate.iloc[-1] == report.probabilities.iloc[-1]  # probabilities are never 0.5 here
    assert skill_established(report) == track_record_trusts_latest
    assert skill_established(report) == (report.has_skill and n_known >= MIN_SCORED_FOR_SKILL)


def test_sell_and_trim_rules():
    over = decide(_inputs(current=0.6))
    assert over.action == "SELL / TRIM" and over.new_position == 0.4  # trimmed to target
    low_p = decide(_inputs(p=0.4, current=0.3))
    assert low_p.action == "SELL / TRIM" and low_p.new_position == 0.0
    high_vol = decide(_inputs(regime="high", dd_kelly=0.2, current=0.35))
    assert high_vol.action == "SELL / TRIM" and high_vol.new_position == 0.2
    assert not high_vol.clauses["BUY: regime is not high-vol"]
    assert decide(_inputs(regime="high", dd_kelly=0.9, current=0.35)).action == "HOLD"  # cap not binding


def test_causal_skill_gate_waits_for_known_outcomes():
    index = pd.bdate_range("2020-01-01", periods=900)
    rng = np.random.default_rng(2)
    labels = pd.Series((rng.random(900) < 0.6).astype(float), index=index)
    good = labels * 0.8 + 0.1  # informative probabilities
    climatology = pd.Series(0.6, index=index)
    gate = causal_skill_gate(good, labels, climatology, horizon=21)
    first_trusted = int(np.argmax(gate.to_numpy() != 0.5))
    # prediction j's outcome is known at the close of j + 21, so 252 outcomes are first known at index 252 + 21 - 1
    assert first_trusted == MIN_SCORED_FOR_SKILL + 21 - 1
    assert (gate.iloc[first_trusted:] == good.iloc[first_trusted:]).all()
    bad = 1 - good  # worse than climatology: never trusted
    assert (causal_skill_gate(bad, labels, climatology, horizon=21) == 0.5).all()


def test_rule_track_record_runs_walk_forward():
    prices = make_prices(1300, mu=0.0005, sigma=0.012, seed=8)
    rf = pd.Series(0.0001, index=prices.index)
    always_bullish = pd.Series(0.7, index=prices.index)
    result = walk_forward(rule_strategy(always_bullish, 0.5, 0.10, 0.25), prices, rf)
    assert result.positions.between(0, 1).all()
    neutral = walk_forward(rule_strategy(pd.Series(0.5, index=prices.index), 0.5, 0.10, 0.25), prices, rf)
    assert (neutral.positions == 0).all()  # p = 0.5 can never trigger BUY


def test_fan_percentiles_are_ordered_and_start_at_one():
    prices = make_prices(1000, mu=0.0004, sigma=0.015, seed=6)
    fan = fan_chart_paths(prices, pd.Series(0.0001, index=prices.index), fraction=0.5, n_paths=300)
    assert list(fan.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert (fan.iloc[0] == 1.0).all()
    assert (fan.diff(axis=1).iloc[:, 1:] >= 0).all().all()
    assert len(fan) == 253
    with pytest.raises(KeyError):
        _ = fan["p99"]
