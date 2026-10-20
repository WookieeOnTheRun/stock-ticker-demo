import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from stockapp.kelly import fractional_kelly


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
