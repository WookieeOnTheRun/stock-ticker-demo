import numpy as np
import pandas as pd
import pytest
from fixtures import make_prices

from stockapp.model.probability import (
    HYPER_GRID,
    brier,
    fit_window,
    log_loss,
    reliability_table,
    walk_forward_model,
)


def test_scores_on_known_values():
    p, y = np.array([0.8, 0.3]), np.array([1.0, 0.0])
    assert brier(p, y) == pytest.approx((0.04 + 0.09) / 2)
    assert log_loss(p, y) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)


def test_calibration_on_well_specified_data():
    # labels drawn from a known logistic model; the calibrated output should match observed frequencies
    rng = np.random.default_rng(0)
    n = 8000
    x = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    true_p = 1 / (1 + np.exp(-(1.2 * x["a"] - 0.8 * x["b"])))
    y = pd.Series((rng.random(n) < true_p).astype(float))
    model = fit_window(x.iloc[:6000], y.iloc[:6000], horizon=1)
    p = model.predict(x.iloc[6000:])
    table = reliability_table(p, y.iloc[6000:].to_numpy())
    populated = table[table["count"] >= 100]
    assert len(populated) >= 5
    assert (populated["observed"] - populated["mean_predicted"]).abs().max() < 0.08
    assert brier(p, y.iloc[6000:].to_numpy()) < brier(np.full(2000, y.iloc[:6000].mean()), y.iloc[6000:].to_numpy())


def test_random_walk_model_has_no_skill_and_reports_everything():
    prices = make_prices(1900, mu=0.0, sigma=0.012, seed=31)  # about 7.5 years
    rf = pd.Series(0.0, index=prices.index)
    report = walk_forward_model(prices, rf, horizon=21)
    assert not report.has_skill
    assert report.trials == len(HYPER_GRID) <= 12
    assert report.probabilities.between(0, 1).all()
    assert report.reliability["count"].sum() == report.metrics["Out-of-sample predictions"]
    assert len(report.latest_contributions) == 12  # 14 features minus the two benchmark features
    assert 0.0 <= report.latest_probability <= 1.0


def test_contributions_sum_to_raw_log_odds():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.standard_normal(3000), "b": rng.standard_normal(3000)})
    y = pd.Series((x["a"] + rng.standard_normal(3000) > 0).astype(float))
    model = fit_window(x, y, horizon=1)
    contributions = model.contributions(x.iloc[:5])
    raw = model.model.predict(x.iloc[:5], raw_score=True)
    bias = np.asarray(model.model.predict(x.iloc[:5], pred_contrib=True))[:, -1]
    np.testing.assert_allclose(contributions.sum(axis=1) + bias, raw)
    assert list(contributions.columns) == ["a", "b"]
