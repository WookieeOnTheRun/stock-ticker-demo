from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yfinance as yf
from curl_cffi.requests.exceptions import HTTPError, Timeout
from yfinance.exceptions import YFRateLimitError

from stockapp.providers import yfinance_provider
from stockapp.providers.base import BAR_COLUMNS
from stockapp.providers.errors import DataError, ProviderUnavailable, RateLimited, TickerNotFound
from stockapp.providers.yfinance_provider import MAX_ATTEMPTS, YFinanceProvider, to_actions, to_bars

DATA = Path(__file__).parent / "data"


def _recorded_aapl_history() -> pd.DataFrame:
    """Ticker('AAPL').history(start='2020-08-25', end='2020-09-03', auto_adjust=False, actions=True), recorded."""
    raw = pd.read_csv(DATA / "aapl_history_2020-08.csv", index_col="Date")
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert("America/New_York").rename("Date")
    return raw


class _FakeTicker:
    def __init__(self, behaviour, calls):
        self.behaviour = behaviour
        self.calls = calls
        self.history_metadata = {"currency": "USD"}

    def history(self, **kwargs):
        self.calls.append(kwargs)
        return self.behaviour()


@pytest.fixture
def patch_ticker(monkeypatch):
    monkeypatch.setattr(yfinance_provider.time, "sleep", lambda seconds: None)
    calls: list[dict] = []

    def install(behaviour):
        monkeypatch.setattr(yf, "Ticker", lambda ticker: _FakeTicker(behaviour, calls))
        return calls

    return install


def _raise(exc):
    def behaviour():
        raise exc

    return behaviour


def test_errors_are_data_errors_and_ticker_not_found_is_a_lookup_error():
    assert issubclass(TickerNotFound, DataError) and issubclass(TickerNotFound, LookupError)
    assert issubclass(ProviderUnavailable, DataError)
    assert issubclass(RateLimited, DataError)


def test_to_bars_maps_recorded_history_to_schema():
    bars = to_bars(_recorded_aapl_history())
    assert list(bars.columns) == BAR_COLUMNS
    assert bars.index.name == "Date" and bars.index.tz is None
    assert bars.index[0] == pd.Timestamp("2020-08-25")  # exchange-local date, not shifted by the UTC offset
    assert all(bars[c].dtype == "float64" for c in BAR_COLUMNS)
    assert bars.loc["2020-08-25", "adj_close"] < bars.loc["2020-08-25", "close"]  # Adj Close also removes dividends


def test_to_actions_extracts_recorded_split():
    actions = to_actions(_recorded_aapl_history())
    assert actions.to_dict("records") == [{"date": pd.Timestamp("2020-08-31"), "type": "split", "value": 4.0}]


def test_to_bars_drops_rows_without_close_and_duplicate_dates():
    raw = _recorded_aapl_history()
    raw.loc[raw.index[-1], ["Close", "Adj Close"]] = float("nan")  # in-progress session: OHLV but no close
    raw = pd.concat([raw, raw.iloc[[0]]])
    bars = to_bars(raw)
    assert len(bars) == len(raw) - 2
    assert bars.index.is_unique and bars.index.is_monotonic_increasing


def test_empty_result_raises_ticker_not_found(patch_ticker):
    calls = patch_ticker(lambda: pd.DataFrame())
    with pytest.raises(TickerNotFound):
        YFinanceProvider().get_daily_bars("MSFT", None, None)
    assert len(calls) == 1  # not retried


def test_timeout_raises_provider_unavailable_after_retries(patch_ticker):
    calls = patch_ticker(_raise(Timeout("Operation timed out after 10000 milliseconds")))
    with pytest.raises(ProviderUnavailable):
        YFinanceProvider().get_daily_bars("MSFT", None, None)
    assert len(calls) == MAX_ATTEMPTS


def test_http_404_raises_ticker_not_found(patch_ticker):
    patch_ticker(_raise(HTTPError("HTTP Error 404: ", response=SimpleNamespace(status_code=404))))
    with pytest.raises(TickerNotFound):
        YFinanceProvider().get_daily_bars("ZZZZ", None, None)


@pytest.mark.parametrize(
    "exc",
    [YFRateLimitError(), HTTPError("HTTP Error 429: ", response=SimpleNamespace(status_code=429))],
)
def test_rate_limit_raises_rate_limited(patch_ticker, exc):
    calls = patch_ticker(_raise(exc))
    with pytest.raises(RateLimited):
        YFinanceProvider().get_daily_bars("MSFT", None, None)
    assert len(calls) == MAX_ATTEMPTS


def test_transient_failure_then_success_is_retried(patch_ticker):
    outcomes = [Timeout("slow"), None]

    def behaviour():
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome
        return _recorded_aapl_history()

    calls = patch_ticker(behaviour)
    provider = YFinanceProvider()
    bars = provider.get_daily_bars("AAPL", date(2020, 8, 25), date(2020, 9, 2))
    assert len(calls) == 2
    assert len(bars) == 7
    assert calls[-1]["end"] == "2020-09-03"  # inclusive end date -> yfinance's exclusive end
    assert provider.get_currency("AAPL") == "USD"
    assert provider.get_corporate_actions("AAPL")["type"].tolist() == ["split"]
    assert len(calls) == 2  # currency and actions came from the same response


def test_unknown_isin_like_symbol_raises_ticker_not_found(monkeypatch):
    # yfinance 1.7.0 raises this from the Ticker constructor for 12-character alphanumeric symbols
    constructed = []

    def failing_ticker(symbol):
        constructed.append(symbol)
        raise ValueError(f"Invalid ISIN number: {symbol}")

    monkeypatch.setattr(yf, "Ticker", failing_ticker)
    with pytest.raises(TickerNotFound):
        YFinanceProvider().get_daily_bars("ZZZZNOTREAL1", None, None)
    assert constructed == ["ZZZZNOTREAL1"]  # mapped, not retried
