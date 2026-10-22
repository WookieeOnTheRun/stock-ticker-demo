import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from stockapp.estimation import ledoit_wolf, make_spd, shrunk_means
from stockapp.optimize import Allocation, ConstrainedKelly, constrained_kelly, hrp_weights, unconstrained_kelly
from stockapp.portfolio import CONSTRAINED, EQUAL, HRP, analyze_portfolio
from stockapp.portfolio_bootstrap import bootstrap_weights
from stockapp.portfolio_data import InsufficientDataError, ReturnsData, default_frequency, returns_matrix
from stockapp.portfolio_eval import fixed_weight_growth
from stockapp.portfolio_input import WatchlistError, parse_ticker_text, parse_watchlist_csv
from stockapp.trades import trade_plan


def _returns(n_obs=1500, n_assets=6, seed=0):
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((n_obs, 1)) * 0.01
    loadings = rng.uniform(0.5, 1.5, (1, n_assets))
    data = factor @ loadings + rng.standard_normal((n_obs, n_assets)) * 0.012 + 0.0004
    return pd.DataFrame(
        data, columns=[f"A{i}" for i in range(n_assets)], index=pd.bdate_range("2018-01-01", periods=n_obs)
    )


def test_unconstrained_kelly_two_uncorrelated_assets():
    mu = np.array([0.06, 0.03])
    sigma = np.diag([0.04, 0.09])
    np.testing.assert_allclose(unconstrained_kelly(mu, sigma), [0.06 / 0.04, 0.03 / 0.09])


@pytest.mark.parametrize(
    ("w_max", "gross_max", "caps"),
    [(0.25, 1.0, None), (0.5, 0.6, None), (0.3, 1.0, {"tech": 0.35}), (0.15, 0.8, {"tech": 0.2, "energy": 0.1})],
)
def test_constrained_solution_satisfies_constraints(w_max, gross_max, caps):
    returns = _returns()
    mu = returns.mean().to_numpy() * 252 * 3
    sigma = returns.cov().to_numpy() * 252
    sectors = ["tech", "tech", "tech", "energy", "energy", None]
    result = constrained_kelly(mu, sigma, 1.0, w_max=w_max, gross_max=gross_max, sector_caps=caps, sectors=sectors)
    w = result.weights
    assert not result.failed
    assert (w >= -1e-6).all()
    assert (w <= w_max + 1e-6).all()
    assert w.sum() <= gross_max + 1e-6
    assert result.cash == pytest.approx(1 - w.sum())
    for sector, cap in (caps or {}).items():
        assert sum(w[i] for i, s in enumerate(sectors) if s == sector) <= cap + 1e-6


def test_loose_constraints_match_scaled_unconstrained():
    mu = np.array([0.04, 0.03, 0.05])
    sigma = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    c = 0.5
    expected = c * unconstrained_kelly(mu, sigma)
    assert (expected > 0).all() and expected.sum() < 1  # long-only feasible, inside the budget
    result = constrained_kelly(mu, sigma, c, w_max=1.0, gross_max=1.0)
    np.testing.assert_allclose(result.weights, expected, atol=1e-6)


def test_solver_error_twice_gives_no_allocation(monkeypatch):
    solver = ConstrainedKelly(4)
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(solver._problem, "solve", boom)
    result = solver.solve(np.full(4, 0.05), np.eye(4) * 0.04, 0.5)
    assert result.failed
    assert result.weights is None and result.cash is None and result.status == "failed"
    assert len(calls) == 2  # the first attempt and one retry
    assert "solver crashed" in result.reason


def test_non_optimal_status_gives_no_allocation(monkeypatch):
    solver = ConstrainedKelly(3, gross_max=0.6)
    monkeypatch.setattr(solver._problem, "solve", lambda *args, **kwargs: None)  # status never becomes optimal
    result = solver.solve(np.full(3, 0.05), np.eye(3) * 0.04, 0.5)
    assert result.failed and result.weights is None
    assert "not optimal" in result.reason


def test_retry_recovers_from_a_failed_first_attempt(monkeypatch):
    mu = np.array([0.04, 0.03, 0.05])
    sigma = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    solver = ConstrainedKelly(3, w_max=1.0, gross_max=1.0)
    real_solve = solver._problem.solve
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(solver._problem, "solve", flaky)
    result = solver.solve(mu, sigma, 0.5)
    assert not result.failed and len(calls) == 2
    # The retry's jitter (1e-6 x mean variance) barely moves the answer.
    np.testing.assert_allclose(result.weights, 0.5 * unconstrained_kelly(mu, sigma), atol=1e-5)


def test_retry_regularizes_a_singular_covariance():
    singular = np.array([[0.04, 0.04], [0.04, 0.04]])  # two identical assets: Cholesky fails on the first attempt
    result = constrained_kelly(np.array([0.05, 0.05]), singular, 0.5, w_max=0.4, gross_max=1.0)
    assert not result.failed
    assert (result.weights >= -1e-6).all() and (result.weights <= 0.4 + 1e-6).all()


def test_inaccurate_solution_is_used_only_if_it_meets_the_constraints(monkeypatch):
    solver = ConstrainedKelly(3, w_max=0.4, gross_max=0.9)

    def inaccurate(weights):
        def fake_solve(*args, **kwargs):
            solver._w.value = np.asarray(weights, dtype=float)
            solver._problem._status = cp.OPTIMAL_INACCURATE

        return fake_solve

    monkeypatch.setattr(solver._problem, "solve", inaccurate([0.3, 0.3, 0.2]))
    accepted = solver.solve(np.full(3, 0.05), np.eye(3) * 0.04, 0.5)
    assert not accepted.failed and accepted.status == "optimal_inaccurate"
    np.testing.assert_allclose(accepted.weights, [0.3, 0.3, 0.2])

    monkeypatch.setattr(solver._problem, "solve", inaccurate([0.5, 0.3, 0.1]))  # breaks w_max = 0.4
    rejected = solver.solve(np.full(3, 0.05), np.eye(3) * 0.04, 0.5)
    assert rejected.failed and "breaks a constraint" in rejected.reason


class _StubSolver:
    """Stands in for ConstrainedKelly in the bootstrap: fails on every `fail_every`-th call, else weights 0.1 each."""

    def __init__(self, n, fail_every):
        self.n, self.fail_every, self.calls = n, fail_every, 0

    def solve(self, mu, sigma, c):
        self.calls += 1
        if self.calls % self.fail_every == 0:
            return Allocation(None, None, "failed", "stub failure")
        return Allocation(np.full(self.n, 0.1), 1 - 0.1 * self.n, "optimal", None)


def test_bootstrap_leaves_failed_resamples_out_of_the_bands():
    excess = _returns(n_obs=300, n_assets=3)
    bands = bootstrap_weights(excess, 0.10, 252, _StubSolver(3, fail_every=3), 0.5, n_resamples=30, time_budget_s=1e9)
    assert (bands.resamples, bands.failures, bands.solved) == (30, 10, 20)
    np.testing.assert_allclose(bands.p5, 0.1)  # only the solved resamples' weights
    np.testing.assert_allclose(bands.p95, 0.1)


def test_bootstrap_bands_are_nan_when_every_resample_fails():
    excess = _returns(n_obs=300, n_assets=3)
    bands = bootstrap_weights(excess, 0.10, 252, _StubSolver(3, fail_every=1), 0.5, n_resamples=30, time_budget_s=1e9)
    assert bands.solved == 0 and bands.failures == bands.resamples
    assert bands.p5.isna().all() and bands.p95.isna().all()


def _returns_data(n_obs=600, n_assets=4):
    returns = _returns(n_obs=n_obs, n_assets=n_assets)
    rf = pd.Series(0.0001, index=returns.index, name="rf")
    return ReturnsData(returns, returns.sub(rf, axis=0), rf, len(returns), 252)


def test_failed_optimizer_leaves_constrained_kelly_out_everywhere(monkeypatch):
    failure = Allocation(None, None, "failed", "stub failure")
    monkeypatch.setattr(ConstrainedKelly, "solve", lambda self, mu, sigma, c: failure)
    analysis = analyze_portfolio(_returns_data(), 0.10, 0.5)
    assert analysis.allocation.failed
    assert analysis.weights[CONSTRAINED].isna().all()
    assert analysis.weights[[EQUAL, HRP]].notna().all().all()  # the other methods are unaffected
    assert CONSTRAINED not in analysis.marks and EQUAL in analysis.marks
    assert CONSTRAINED not in analysis.split_growth.columns and EQUAL in analysis.split_growth.columns
    assert analysis.split_failure == "stub failure"
    assert analysis.frontier.empty and list(analysis.frontier.columns) == ["c", "volatility", "excess_return"]
    assert analysis.bands.solved == 0 and analysis.bands.p5.isna().all()


def test_successful_optimizer_is_in_every_output():
    analysis = analyze_portfolio(_returns_data(), 0.10, 0.5)
    assert not analysis.allocation.failed and analysis.split_failure is None
    assert analysis.weights[CONSTRAINED].notna().all()
    assert CONSTRAINED in analysis.marks and CONSTRAINED in analysis.split_growth.columns
    assert not analysis.frontier.empty and analysis.bands.failures == 0


def test_ledoit_wolf_is_spd_and_better_conditioned():
    returns = _returns(n_obs=120, n_assets=30, seed=2)  # few observations per asset: a poorly conditioned sample
    estimate = ledoit_wolf(returns)
    np.testing.assert_allclose(estimate.sigma, estimate.sigma.T)
    np.linalg.cholesky(estimate.sigma)
    assert estimate.condition_shrunk <= estimate.condition_sample
    assert 0 < estimate.shrinkage <= 1


def test_make_spd_adds_smallest_working_jitter():
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    fixed, eps = make_spd(singular)
    assert eps > 0
    np.linalg.cholesky(fixed)
    already, eps0 = make_spd(np.eye(2))
    assert eps0 == 0 and np.allclose(already, np.eye(2))


def test_shrunk_means_follow_tier0_posterior():
    returns = _returns(n_obs=500)
    shrunk = shrunk_means(returns, 0.10, 252)
    tau2 = (0.10 / 252) ** 2
    se2 = returns.var() / 500
    np.testing.assert_allclose(shrunk, returns.mean() * tau2 / (tau2 + se2))
    assert (shrunk.abs() <= returns.mean().abs()).all()


def test_hrp_weights_and_identical_assets():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(2000) * 0.01
    b = rng.standard_normal(2000) * 0.01
    returns = pd.DataFrame({"X1": a, "X2": a, "Y": b})
    hrp = hrp_weights(returns)
    assert (hrp.weights >= 0).all()
    assert hrp.weights.sum() == pytest.approx(1)
    assert hrp.weights["Y"] == pytest.approx(0.5, abs=0.02)
    general = hrp_weights(_returns())
    assert general.weights.sum() == pytest.approx(1) and (general.weights >= 0).all()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"ticker,weight,notes\nMSFT,0.1,x\n", "Unexpected column"),
        (("ticker\n" + "\n".join(f"T{i}" for i in range(51))).encode(), "At most 50"),
        (b"ticker\n" + b"MSFT\n" * 300_000, "larger than"),
        (b"ticker\nMSFT\n=HYPERLINK(1)\n", "Invalid ticker"),
        (b"ticker,weight\nMSFT,0.8\nAAPL,0.5\n", "sum to"),
    ],
)
def test_csv_upload_rejections(data, message):
    with pytest.raises(WatchlistError, match=message):
        parse_watchlist_csv(data)


def test_csv_upload_accepts_valid_files():
    weights = parse_watchlist_csv(b"Ticker,Weight\nmsft,0.3\nAAPL,0.2\n")
    assert weights.tickers == ["MSFT", "AAPL"] and weights.weights == {"MSFT": 0.3, "AAPL": 0.2}
    shares = parse_watchlist_csv(b"ticker,shares\nVOD.L,100\n")
    assert shares.shares == {"VOD.L": 100.0}
    assert parse_ticker_text("msft, aapl\nmsft") == ["MSFT", "AAPL"]


def test_weekly_returns_are_compounded():
    index = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(5)
    prices = {t: pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 300)), index=index) for t in ("A", "B")}
    rf = pd.Series(0.0001, index=index)
    weekly = returns_matrix(prices, rf, "Weekly")
    daily = returns_matrix(prices, rf, "Daily").returns
    week = weekly.returns.index[3]
    in_week = daily.loc[(daily.index > weekly.returns.index[2]) & (daily.index <= week), "A"]
    assert weekly.returns.loc[week, "A"] == pytest.approx(np.prod(1 + in_week) - 1)
    assert weekly.returns.loc[week, "A"] != pytest.approx(in_week.sum())
    assert weekly.rf.loc[week] == pytest.approx((1.0001) ** len(in_week) - 1)
    assert weekly.periods_per_year == 52


def test_returns_matrix_requires_a_year_of_common_dates():
    index = pd.bdate_range("2024-01-01", periods=400)
    prices = {"A": pd.Series(100.0, index=index), "B": pd.Series(50.0, index=index[200:])}
    with pytest.raises(InsufficientDataError, match="200 trading dates"):
        returns_matrix(prices, pd.Series(0.0, index=index))


def test_default_frequency_by_exchange():
    assert default_frequency(["MSFT", "AAPL", "BRK-B"]) == "Daily"
    assert default_frequency(["MSFT", "VOD.L"]) == "Weekly"


def test_trade_plan_turnover_and_shares():
    plan = trade_plan(
        {"A": 0.3}, {"A": 0.2, "B": 0.3}, cost_bps=10, latest_prices={"A": 50.0, "B": 20.0}, portfolio_value=10_000
    )
    assert plan.turnover == pytest.approx(0.5 * (0.1 + 0.3 + 0.2))
    assert plan.cost_fraction == pytest.approx(plan.turnover * 0.001)
    assert plan.table.loc["B", "Trade (shares)"] == pytest.approx(0.3 * 10_000 / 20)
    assert plan.table.loc["Cash", "Target weight"] == pytest.approx(0.5)


def test_fixed_weight_growth_with_cash():
    index = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.10, -0.05, 0.02]}, index=index)
    growth = fixed_weight_growth(returns, pd.Series(0.001, index=index), {"half": pd.Series({"A": 0.5})})
    expected = np.prod([1 + 0.5 * r + 0.5 * 0.001 for r in (0.10, -0.05, 0.02)])
    assert growth["half"].iloc[-1] == pytest.approx(expected)
    assert growth["half"].iloc[0] == 1.0
