import logging
import random
import time
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from curl_cffi.requests.exceptions import HTTPError, RequestException
from yfinance.exceptions import YFRateLimitError, YFTickerMissingError

from stockapp.providers.base import ACTION_COLUMNS, BAR_COLUMNS, Bars
from stockapp.providers.errors import DataError, ProviderUnavailable, RateLimited, TickerNotFound

logger = logging.getLogger(__name__)

TIMEOUT_S = 10  # per HTTP request
MAX_ATTEMPTS = 3  # total tries, including the first
BACKOFF_BASE_S = 0.5  # sleep before retry k (k = 0, 1) is BACKOFF_BASE_S * 2**k, scaled by random jitter in [0.5, 1.5)

# Column names returned by Ticker.history(auto_adjust=False, actions=True) in yfinance 1.7.0. Note that Yahoo's
# "Close" is already split-adjusted (not raw); "Adj Close" is additionally dividend-adjusted.
COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def with_retries[T](
    fn: Callable[[], T],
    attempts: int = MAX_ATTEMPTS,
    base_delay: float = BACKOFF_BASE_S,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn, retrying ProviderUnavailable and RateLimited with exponential backoff and jitter."""
    for attempt in range(attempts):
        try:
            return fn()
        except (ProviderUnavailable, RateLimited) as exc:
            if attempt == attempts - 1:
                raise
            delay = base_delay * 2**attempt * random.uniform(0.5, 1.5)  # noqa: S311 - retry jitter, not security
            logger.info(
                "Retrying after %s (attempt %d of %d, sleeping %.2fs)", type(exc).__name__, attempt + 1, attempts, delay
            )
            sleep(delay)
    raise AssertionError("unreachable")


def map_error(ticker: str, exc: Exception) -> DataError | None:
    """Translate a yfinance / curl_cffi exception into a typed DataError, or None if it isn't a data error."""
    if isinstance(exc, YFRateLimitError):
        return RateLimited(f"Rate limited while fetching '{ticker}'")
    if isinstance(exc, YFTickerMissingError):
        return TickerNotFound(f"No price data for '{ticker}'. Check the ticker symbol.")
    if isinstance(exc, HTTPError):
        status = getattr(exc.response, "status_code", None)
        if status == 404:
            return TickerNotFound(f"No price data for '{ticker}'. Check the ticker symbol.")
        if status == 429:
            return RateLimited(f"Rate limited while fetching '{ticker}'")
        return ProviderUnavailable(f"Data provider error (HTTP {status}) while fetching '{ticker}'")
    if isinstance(exc, RequestException | OSError):
        return ProviderUnavailable(f"Could not reach the data provider for '{ticker}' ({type(exc).__name__})")
    return None


def to_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a Ticker.history frame to the Bars schema: exchange-local dates, float columns, cleaned and sorted."""
    index = pd.DatetimeIndex(raw.index)
    if index.tz is not None:
        index = index.tz_localize(None)  # keep the exchange-local wall-clock date
    bars = raw.rename(columns=COLUMN_MAP).reindex(columns=BAR_COLUMNS).astype("float64")
    bars.index = index.normalize().rename("Date")
    bars = bars.dropna(how="all")
    bars = bars.dropna(subset=["close", "adj_close"])  # e.g. an in-progress Asian session with OHLV but no close yet
    bars = bars.sort_index()
    return Bars.validate(bars[~bars.index.duplicated(keep="last")])


def to_actions(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract splits and dividends from a Ticker.history(actions=True) frame."""
    dates = pd.DatetimeIndex(raw.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    parts = []
    for column, kind in (("Stock Splits", "split"), ("Dividends", "dividend")):
        if column in raw:
            values = raw[column].to_numpy(dtype=float)
            mask = values > 0
            parts.append(pd.DataFrame({"date": dates[mask], "type": kind, "value": values[mask]}))
    if not parts:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[s]"), "type": [], "value": []})[ACTION_COLUMNS]
    return pd.concat(parts, ignore_index=True).sort_values(["date", "type"], ignore_index=True)[ACTION_COLUMNS]


class YFinanceProvider:
    """
    yfinance adapter. One Ticker.history request returns bars, splits/dividends and the currency (via
    history_metadata), so those are kept from the latest bars request and reused by get_corporate_actions and
    get_currency without another request.
    """

    name = "yfinance"

    def __init__(self) -> None:
        yf.config.network.retries = 0  # retries are handled here, with jitter
        yf.config.debug.hide_exceptions = False  # surface typed errors instead of an empty frame
        self._actions: dict[str, pd.DataFrame] = {}
        self._currency: dict[str, str] = {}

    def _history(self, ticker: str, **kwargs: Any) -> pd.DataFrame:
        def attempt() -> pd.DataFrame:
            t = yf.Ticker(ticker)
            try:
                raw: pd.DataFrame = t.history(
                    interval="1d", auto_adjust=False, actions=True, timeout=TIMEOUT_S, **kwargs
                )
            except Exception as exc:
                mapped = map_error(ticker, exc)
                if mapped is None:
                    raise
                raise mapped from exc
            if raw is None or raw.empty:
                raise TickerNotFound(f"No price data for '{ticker}'. Check the ticker symbol.")
            # yfinance 1.7.0: the quote currency is in Ticker.history_metadata["currency"] after a history call
            # (the same value fast_info["currency"] returns), so reading it costs no extra request.
            currency = (t.history_metadata or {}).get("currency")
            if currency:
                self._currency[ticker] = str(currency)
            self._actions[ticker] = to_actions(raw)
            return raw

        return with_retries(attempt)

    def get_daily_bars(self, ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"period": "max"} if start is None else {"start": start.isoformat()}
        if end is not None:
            kwargs["end"] = (end + timedelta(days=1)).isoformat()  # yfinance's end is exclusive
        bars = to_bars(self._history(ticker, **kwargs))
        if bars.empty:
            raise TickerNotFound(f"No price data for '{ticker}'. Check the ticker symbol.")
        return bars

    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._actions:
            self._history(ticker, period="max")
        return self._actions[ticker]

    def get_currency(self, ticker: str) -> str:
        if ticker not in self._currency:
            self._history(ticker, period="5d")
        if ticker not in self._currency:
            raise ProviderUnavailable(f"The data provider returned no currency for '{ticker}'")
        return self._currency[ticker]
