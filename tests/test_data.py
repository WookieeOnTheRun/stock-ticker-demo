import numpy as np
import pandas as pd
import pytest
import yfinance as yf
from fixtures import make_prices

from stockapp.data import compute_returns, fetch_price_history, normalize_ticker


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


class _FakeTicker:
    """Stands in for yf.Ticker: history() returns a canned frame, as the yfinance provider calls it."""

    def __init__(self, frame):
        self.frame = frame
        self.history_metadata = {"currency": "USD"}

    def history(self, **kwargs):
        return self.frame


def test_fetch_price_history_empty_raises(monkeypatch):
    monkeypatch.setattr(yf, "Ticker", lambda ticker: _FakeTicker(pd.DataFrame()))
    with pytest.raises(LookupError):
        fetch_price_history("MSFT")


def test_fetch_price_history_cleans_frame(monkeypatch):
    # Shape mirrors yf.Ticker.history(auto_adjust=False): OHLC float columns, Adj Close, int Volume.
    # Shuffled rows, an unnamed index, a NaN close and int closes exercise the cleanup steps.
    close = make_prices(10, mu=0.0005, sigma=0.01, seed=1).round().astype("int64")
    frame = pd.DataFrame(
        {"Close": close, "Adj Close": close, "High": close + 1, "Low": close - 1, "Open": close, "Volume": 1_000}
    )
    frame = frame.astype({"Close": "float64", "Adj Close": "float64"})
    frame.iloc[3, frame.columns.get_loc("Close")] = np.nan
    frame = frame.sample(frac=1, random_state=0)
    frame.index.name = None
    monkeypatch.setattr(yf, "Ticker", lambda ticker: _FakeTicker(frame))

    result = fetch_price_history("MSFT")

    assert result.dtype == np.float64
    assert result.index.is_monotonic_increasing
    assert result.index.name == "Date"
    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) == 9  # NaN close dropped
