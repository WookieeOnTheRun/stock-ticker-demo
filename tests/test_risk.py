import numpy as np
import pandas as pd
import pytest
from fixtures import make_prices

from stockapp import risk


def _returns(n=2000, seed=4, dist="normal"):
    return make_prices(n + 1, mu=0.0004, sigma=0.015, seed=seed, dist=dist).pct_change().dropna()


def test_var_and_cvar_match_numpy_quantile():
    r = _returns()
    q = np.quantile(r.to_numpy(), 0.05)
    assert risk.historical_var(r) == pytest.approx(-q)
    assert risk.historical_cvar(r) == pytest.approx(-r[r <= q].mean())
    assert risk.historical_cvar(r) >= risk.historical_var(r) > 0


def test_cvar_at_least_var_on_fat_tails():
    r = _returns(dist="t")
    for alpha in (0.01, 0.05, 0.10):
        assert risk.historical_cvar(r, alpha) >= risk.historical_var(r, alpha)


def test_parametric_var_matches_normal_formula():
    r = _returns()
    assert risk.parametric_var(r) == pytest.approx(-(r.mean() - 1.6448536269514722 * r.std()))


def test_max_drawdown_on_hand_built_path():
    dates = pd.date_range("2024-01-01", periods=8)
    prices = pd.Series([100, 120, 110, 60, 90, 119, 125, 130.0], index=dates)
    dd = risk.max_drawdown(prices)
    assert dd.depth == pytest.approx(0.5)  # 120 -> 60
    assert dd.peak_date == dates[1]
    assert dd.trough_date == dates[3]
    assert dd.recovery_date == dates[6]  # first close at or above 120
    assert dd.duration_days == 5


def test_max_drawdown_without_recovery_or_drawdown():
    dates = pd.date_range("2024-01-01", periods=4)
    falling = risk.max_drawdown(pd.Series([100, 90, 80, 85.0], index=dates))
    assert falling.recovery_date is None and falling.duration_days == 3
    rising = risk.max_drawdown(pd.Series([1, 2, 3, 4.0], index=dates))
    assert rising.depth == 0 and rising.duration_days == 0


def test_beta_recovers_slope_and_r2_falls_with_noise():
    rng = np.random.default_rng(7)
    bench = pd.Series(rng.normal(0, 0.01, 3000), index=pd.bdate_range("2015-01-01", periods=3000))
    fits = []
    for noise in (0.002, 0.01, 0.03):
        stock = 2 * bench + rng.normal(0, noise, len(bench))
        fits.append(risk.beta_r2(stock, bench))
    assert fits[0].beta == pytest.approx(2, abs=0.05)
    assert fits[0].r2 > fits[1].r2 > fits[2].r2


def test_vol_estimates_are_annualized():
    r = _returns(n=3000)
    assert risk.rolling_vol(r).dropna().median() == pytest.approx(0.015 * np.sqrt(252), rel=0.15)
    assert risk.ewma_vol(r).iloc[-1] == pytest.approx(0.015 * np.sqrt(252), rel=0.4)


def test_omega_sortino_calmar_signs():
    r = _returns()
    rf = pd.Series(0.0, index=r.index)
    assert risk.omega(r) > 0
    assert risk.omega(pd.Series([0.01, 0.02])) == float("inf")
    up = pd.Series(np.linspace(100, 200, 300), index=pd.bdate_range("2020-01-01", periods=300))
    assert risk.calmar(up) == 0.0  # no drawdown
    assert risk.sortino(r, rf) == pytest.approx(r.mean() / np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252))


def test_garch_forecast_on_simulated_garch():
    rng = np.random.default_rng(0)
    n, omega, a, b = 2500, 0.02, 0.08, 0.90
    eps = np.empty(n)
    s2 = omega / (1 - a - b)
    for t in range(n):
        eps[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + a * eps[t] ** 2 + b * s2
    r = pd.Series(eps / 100, index=pd.bdate_range("2015-01-01", periods=n))

    fc = risk.garch_vol_forecast(r, horizon=21)

    assert fc.converged
    assert fc.path.shape == (21,)
    assert 0.5 < fc.params["alpha[1]"] + fc.params["beta[1]"] < 1
    long_run = np.sqrt(omega / (1 - a - b) * 252) / 100
    assert fc.annualized_vol == pytest.approx(long_run, rel=0.5)
