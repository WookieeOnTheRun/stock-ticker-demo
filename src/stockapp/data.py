import re

import numpy as np
import pandas as pd
import yfinance as yf

TICKER_PATTERN = re.compile(r"[A-Z0-9.^=\-]{1,15}")
HISTORY_PERIOD = "5y"  # ~1250 trading days; covers the max lookback slider value and all Kelly windows


def normalize_ticker(raw: str) -> str | None:
    """Return the upper-cased ticker, or None if it contains unexpected characters."""
    ticker = raw.strip().upper()
    return ticker if TICKER_PATTERN.fullmatch(ticker) else None


def fetch_price_history(ticker: str) -> pd.Series:
    """
    Fetch split/dividend-adjusted daily closes for the last HISTORY_PERIOD.
    Uses yfinance (free, no API key). Returns a float Series named "Close" indexed by Date.
    Raises LookupError if no data is returned, so empty results are never cached.
    """
    df = yf.download(ticker, period=HISTORY_PERIOD, progress=False, auto_adjust=True, multi_level_index=False)
    if df.empty or "Close" not in df:
        raise LookupError(f"No price data returned for '{ticker}'. Check the ticker symbol.")
    close: pd.Series = df["Close"].dropna().astype(float).sort_index()
    close.index = pd.to_datetime(close.index)
    close.index.name = "Date"
    return close


def compute_returns(close: pd.Series) -> pd.DataFrame:
    """
    Compute daily simple and log returns from adjusted closes.
    Returns DataFrame with columns: [simple_return, log_return].
    """
    simple = close.pct_change().dropna()
    return pd.DataFrame({"simple_return": simple, "log_return": np.log1p(simple)})
