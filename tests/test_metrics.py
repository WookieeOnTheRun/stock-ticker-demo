import pandas as pd
import pytest

from stockapp.metrics import cagr, growth_rate, sharpe_ratio


def test_cagr_of_doubling_over_one_year():
    close = pd.Series([100.0] * 252 + [200.0])  # 252 daily returns
    assert cagr(close) == pytest.approx(1.0)


def test_sharpe_and_growth_formulas():
    e = pd.Series([0.01, -0.005, 0.002, 0.004])
    rf = pd.Series([0.0001] * 4)
    assert sharpe_ratio(e) == pytest.approx(e.mean() / e.std() * 252**0.5)
    assert growth_rate(0.5, rf, e) == pytest.approx(252 * (0.0001 + 0.5 * e.mean() - 0.25 * e.var() / 2))
