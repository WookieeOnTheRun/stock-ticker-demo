"""Provider-neutral market data schema (Bars) and the PriceProvider protocol."""

from datetime import date
from typing import Protocol

import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
ACTION_COLUMNS = ["date", "type", "value"]
ACTION_TYPES = {"split", "dividend"}


class Bars:
    """
    Schema for daily bars: a DataFrame with a timezone-naive DatetimeIndex named "Date" holding the
    exchange-local trading date, and float columns BAR_COLUMNS (open, high, low, close, adj_close, volume).
    """

    columns = BAR_COLUMNS

    @staticmethod
    def empty() -> pd.DataFrame:
        """An empty frame in the Bars schema."""
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=pd.DatetimeIndex([], name="Date")
        )

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """Raise ValueError if df does not follow the schema; return it unchanged otherwise."""
        if list(df.columns) != BAR_COLUMNS:
            raise ValueError(f"Bars columns must be {BAR_COLUMNS}, got {list(df.columns)}")
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.name != "Date" or df.index.tz is not None:
            raise ValueError("Bars index must be a timezone-naive DatetimeIndex named 'Date'")
        if not all(pd.api.types.is_float_dtype(df[c]) for c in BAR_COLUMNS):
            raise ValueError("Bars columns must all be float")
        return df


class PriceProvider(Protocol):
    """Interface every market data source implements; `name` keys the source's rows in the SQLite cache."""

    name: str

    def get_daily_bars(self, ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
        """Daily bars (Bars schema) for start..end inclusive; None means earliest available / latest."""
        ...

    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        """Splits and dividends: columns date (datetime64), type ("split" | "dividend"), value (ratio or amount)."""
        ...

    def get_currency(self, ticker: str) -> str:
        """Quote currency code as reported by the provider (may be a minor unit such as "GBp")."""
        ...
