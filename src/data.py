import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np


def fetch_price_history(ticker: str, period_days: int) -> pd.DataFrame:
    """
    Fetch adjusted close price history for a ticker over the last `period_days` days.
    Uses yfinance (free, no API key).
    Returns DataFrame with columns: [Date, Adj Close] indexed by Date.
    """
    if period_days <= 0:
        period_days = 365
    end = datetime.utcnow()
    start = end - timedelta(days=period_days + 30)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame(columns=["Date", "Close"]).set_index("Date")
    out = df[["Close"]].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.tail(period_days)  # limit to requested window
    out.index.name = "Date"
    return out


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple and log returns from adjusted close.
    Returns DataFrame with columns: [simple_return, log_return].
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=["simple_return", "log_return"]) 
    adj = price_df["Close"].astype(float)
    simple = adj.pct_change().dropna()
    ratio = (adj / adj.shift(1)).dropna()
    logret = np.log(ratio)
    returns = pd.DataFrame({
        "simple_return": simple,
        "log_return": logret
    })
    returns.index.name = "Date"
    return returns
