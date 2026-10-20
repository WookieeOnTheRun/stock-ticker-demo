import re

import numpy as np
import pandas as pd

TICKER_PATTERN = re.compile(r"[A-Z0-9.^=\-]{1,15}")


def normalize_ticker(raw: str) -> str | None:
    """Return the upper-cased ticker, or None if it contains unexpected characters."""
    ticker = raw.strip().upper()
    return ticker if TICKER_PATTERN.fullmatch(ticker) else None


def fetch_price_history(ticker: str) -> pd.Series:
    """
    Split/dividend-adjusted daily closes for the last 10 years, from the cached data layer
    (stockapp.repository.load_bars). Returns a float Series named "Close" indexed by Date.
    Raises TickerNotFound (a LookupError) if no data is returned, so empty results are never cached.
    """
    from stockapp.repository import load_bars  # deferred: repository imports normalize_ticker from this module

    close = load_bars(ticker)["adj_close"].rename("Close")
    close.index.name = "Date"
    return close


def compute_returns(close: pd.Series) -> pd.DataFrame:
    """
    Compute daily simple and log returns from adjusted closes.
    Returns DataFrame with columns: [simple_return, log_return].
    """
    simple = close.pct_change().dropna()
    return pd.DataFrame({"simple_return": simple, "log_return": np.log1p(simple)})
