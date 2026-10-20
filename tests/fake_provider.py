from datetime import date

import numpy as np
import pandas as pd

from stockapp.providers.base import BAR_COLUMNS, Bars
from stockapp.providers.errors import ProviderUnavailable, TickerNotFound


class FakeProvider:
    """
    Deterministic in-memory PriceProvider. Bars are a seeded random walk on business days ending `end`.
    Records every get_daily_bars call as (ticker, start, end); set `fail = True` to simulate an outage.
    """

    name = "fake"

    def __init__(self, n: int = 600, end: date = date(2026, 10, 16), seed: int = 1, currency: str = "USD") -> None:
        rng = np.random.default_rng(seed)
        index = pd.bdate_range(end=end, periods=n, name="Date")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
        self.bars = self._frame(index, close)
        self.actions = pd.DataFrame(
            {"date": [index[n // 6], index[n // 2]], "type": ["split", "dividend"], "value": [2.0, 0.5]}
        )
        self.currency = currency
        self.calls: list[tuple[str, date | None, date | None]] = []
        self.fail = False
        self._rng = rng

    @staticmethod
    def _frame(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "adj_close": close * 0.98},
            index=index,
        )
        frame["volume"] = 1e6
        return Bars.validate(frame[BAR_COLUMNS].astype("float64"))

    def append_days(self, k: int) -> None:
        """Add k new business days after the last bar."""
        index = pd.bdate_range(self.bars.index[-1] + pd.offsets.BDay(), periods=k, name="Date")
        close = self.bars["close"].iloc[-1] * np.exp(np.cumsum(self._rng.normal(0.0004, 0.02, k)))
        self.bars = pd.concat([self.bars, self._frame(index, close)])

    def get_daily_bars(self, ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
        self.calls.append((ticker, start, end))
        if self.fail:
            raise ProviderUnavailable("simulated outage")
        bars = self.bars
        if start is not None:
            bars = bars[bars.index >= pd.Timestamp(start)]
        if end is not None:
            bars = bars[bars.index <= pd.Timestamp(end)]
        if bars.empty:
            raise TickerNotFound(f"No price data for '{ticker}'")
        return bars.copy()

    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        return self.actions.copy()

    def get_currency(self, ticker: str) -> str:
        return self.currency
