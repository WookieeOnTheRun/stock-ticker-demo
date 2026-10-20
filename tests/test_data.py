import numpy as np
import pandas as pd

from data import compute_returns, normalize_ticker


def test_normalize_ticker():
    assert normalize_ticker("  msft ") == "MSFT"
    assert normalize_ticker("brk-b") == "BRK-B"
    assert normalize_ticker("^gspc") == "^GSPC"
    assert normalize_ticker("") is None
    assert normalize_ticker("../etc/passwd") is None
    assert normalize_ticker("A" * 16) is None


def test_compute_returns():
    close = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2024-01-01", periods=3, name="Date"))
    returns = compute_returns(close)
    assert list(returns.columns) == ["simple_return", "log_return"]
    assert returns.index.name == "Date"
    np.testing.assert_allclose(returns["simple_return"], [0.1, -0.1])
    np.testing.assert_allclose(returns["log_return"], np.log([1.1, 0.9]))
