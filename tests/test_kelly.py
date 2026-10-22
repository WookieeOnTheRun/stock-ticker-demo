import pandas as pd
import pytest
from fixtures import make_prices
from hypothesis import given
from hypothesis import strategies as st

from stockapp.kelly import fractional_kelly, log_optimal_kelly, shrunk_mean


def test_too_few_returns_gives_zero():
    kr = fractional_kelly(pd.Series([0.01]))
    assert kr.raw_kelly == 0.0
    assert kr.applied_kelly == 0.0


def test_zero_variance_gives_zero():
    kr = fractional_kelly(pd.Series([0.01] * 10))
    assert kr.raw_kelly == 0.0
    assert kr.applied_kelly == 0.0


def test_negative_mean_clamped_to_zero():
    kr = fractional_kelly(pd.Series([-0.01, -0.02, 0.005, -0.015]))
    assert kr.raw_kelly < 0
    assert kr.applied_kelly == 0.0


def test_fraction_applied_and_clamped_to_max():
    returns = pd.Series([0.01, 0.02, 0.015, 0.005])
    kr = fractional_kelly(returns, k_fraction=0.5)
    assert kr.mean_return == pytest.approx(returns.mean())
    assert kr.variance == pytest.approx(returns.var())
    assert kr.raw_kelly == pytest.approx(returns.mean() / returns.var())
    assert kr.applied_kelly == 1.0  # raw * 0.5 far exceeds the 1.0 clamp


def test_fraction_within_bounds():
    returns = pd.Series([0.05, -0.05, 0.03, -0.03, 0.002])
    kr = fractional_kelly(returns, k_fraction=0.5)
    assert kr.applied_kelly == pytest.approx(0.5 * returns.mean() / returns.var())


# Extreme magnitudes (~1e308) overflow the variance; the clamp must still hold.
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@given(
    returns=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2),
    k_fraction=st.floats(min_value=0.0, max_value=1.0),
)
def test_applied_kelly_always_within_unit_interval(returns, k_fraction):
    kr = fractional_kelly(pd.Series(returns), k_fraction=k_fraction)
    assert 0.0 <= kr.applied_kelly <= 1.0


def _simple_returns(n, mu, sigma, seed, dist="normal"):
    r = make_prices(n + 1, mu=mu, sigma=sigma, seed=seed, dist=dist).pct_change().dropna()
    return r, pd.Series(0.0, index=r.index)


def test_gaussian_and_log_optimal_agree_on_gaussian_data():
    # n=20000, sigma=0.01: the mean's standard error is ~7% of mu, so f* = mu/sigma^2 = 10 is well identified.
    r, rf = _simple_returns(20000, mu=0.001, sigma=0.01, seed=11)
    gaussian = fractional_kelly(r, rf=rf).raw_kelly
    assert gaussian == pytest.approx(10, rel=0.25)
    assert log_optimal_kelly(r, rf) == pytest.approx(gaussian, rel=0.10)


def test_log_optimal_below_gaussian_on_fat_tails():
    r, rf = _simple_returns(20000, mu=0.001, sigma=0.01, seed=11, dist="t")
    assert log_optimal_kelly(r, rf) < fractional_kelly(r, rf=rf).raw_kelly


def test_shrinkage_never_increases_magnitude():
    r, _ = _simple_returns(500, mu=0.0005, sigma=0.02, seed=5)
    for prior in (0.01, 0.1, 0.5):
        assert abs(shrunk_mean(r, prior)) <= abs(r.mean())


def test_shrinkage_strengthens_as_sample_shrinks():
    r, _ = _simple_returns(5000, mu=0.0005, sigma=0.02, seed=5)
    kept = [shrunk_mean(r.tail(n), 0.1) / r.tail(n).mean() for n in (100, 1000, 5000)]
    assert kept[0] < kept[1] < kept[2]


def test_shrinkage_vanishes_with_infinite_prior():
    r, _ = _simple_returns(500, mu=0.0005, sigma=0.02, seed=5)
    assert shrunk_mean(r, 1e12) == pytest.approx(r.mean(), rel=1e-12)
