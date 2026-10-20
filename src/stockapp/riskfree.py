import pandas as pd

from stockapp.repository import load_bars

RF_TICKER = "^IRX"  # 13-week US T-bill yield, quoted in annualized percent (e.g. 4.08 means 4.08%)
TRADING_DAYS = 252
FFILL_LIMIT = 5  # max consecutive missing rf days carried forward (about one trading week)


def fetch_rf_annual_pct(years: int | None = 10) -> pd.Series:
    """
    Fetch the 13-week US T-bill yield (^IRX) for the last `years` years (None = all), in annualized percent.
    Uses the same cached data layer as fetch_price_history; raises TickerNotFound if nothing comes back.
    """
    return load_bars(RF_TICKER, years)["adj_close"].rename("Close")


def daily_rf(annual_pct: pd.Series) -> pd.Series:
    """
    Convert an annualized percent yield to a daily rate: d = (1 + y/100) ** (1/252) - 1.

    ^IRX is a discount-basis T-bill yield. Treating it as an effective annual rate is a small,
    documented approximation: it understates the effective rate by about 0.17 percentage points at
    a 4% yield (0.05 at 2%, 0.26 at 5.35%).
    """
    return (1 + annual_pct / 100) ** (1 / TRADING_DAYS) - 1


def align_rf(rf: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """
    Align a daily rf series to the stock's trading dates.

    rf observations are forward-filled onto the stock's dates for at most FFILL_LIMIT steps and never
    back-filled, since that would leak future rates into the past. Stock dates before the first rf
    observation, and dates whose most recent rf value is too stale to fill, are dropped.
    """
    combined = rf.reindex(rf.index.union(index)).sort_index().ffill(limit=FFILL_LIMIT)
    return combined.reindex(index).dropna()
