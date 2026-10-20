from dataclasses import dataclass
from typing import Literal

import pandas as pd

from stockapp.kelly import TRADING_DAYS

Frequency = Literal["Daily", "Weekly"]
MIN_COMMON_DAYS = 252
WEEKS_PER_YEAR = 52
WEEK_END = "W-FRI"


class InsufficientDataError(ValueError):
    """Too few common observations to estimate means and covariances; the message explains why."""


def exchange_suffix(ticker: str) -> str:
    """Exchange suffix of a ticker ('.L' for VOD.L); '' for US listings, indices and FX."""
    if ticker.startswith("^") or "=" in ticker or "." not in ticker:
        return ""
    return ticker[ticker.rfind(".") :]


def default_frequency(tickers: list[str]) -> Frequency:
    """
    Weekly when the tickers trade on exchanges with different closing times (different suffixes), Daily otherwise.
    Daily returns of assets that close hours apart are measured over different windows, which biases their
    correlation toward zero; weekly returns mostly wash that out.
    """
    return "Weekly" if len({exchange_suffix(t) for t in tickers}) > 1 else "Daily"


@dataclass(frozen=True)
class ReturnsData:
    returns: pd.DataFrame
    """Asset returns per period (columns = tickers) on common dates."""
    excess: pd.DataFrame
    """Returns minus the risk-free rate over the same period."""
    rf: pd.Series
    common_days: int
    """Trading dates shared by every asset (before any weekly resampling)."""
    periods_per_year: float


def returns_matrix(prices: dict[str, pd.Series], rf_daily: pd.Series, frequency: Frequency = "Daily") -> ReturnsData:
    """
    Returns matrix on the intersection of the assets' trading dates. Weekly returns are computed from week-end
    (Friday-labelled) prices, so they compound the daily returns rather than summing them; the risk-free rate is
    compounded over the same weeks. Annualization uses 252 periods (Daily) or 52 (Weekly).
    Raises InsufficientDataError below MIN_COMMON_DAYS common trading dates.
    """
    joined = pd.concat(prices, axis=1, join="inner").dropna()
    rf_aligned = rf_daily.reindex(joined.index)
    joined = joined.loc[rf_aligned.notna().to_numpy()]
    rf_aligned = rf_aligned.dropna()
    common = len(joined)
    if common < MIN_COMMON_DAYS:
        raise InsufficientDataError(
            f"Only {common} trading dates are common to all tickers; at least {MIN_COMMON_DAYS} (about a year) are "
            "needed for usable means and covariances. Remove the ticker with the shortest history or load more."
        )
    if frequency == "Weekly":
        week_prices = joined.resample(WEEK_END).last().dropna()
        returns = week_prices.pct_change().iloc[1:]
        rf = ((1 + rf_aligned).resample(WEEK_END).prod() - 1).reindex(returns.index).fillna(0.0)
        periods: float = WEEKS_PER_YEAR
    else:
        returns = joined.pct_change().iloc[1:]
        rf = rf_aligned.reindex(returns.index)
        periods = TRADING_DAYS
    return ReturnsData(
        returns=returns,
        excess=returns.sub(rf, axis=0),
        rf=rf.rename("rf"),
        common_days=common,
        periods_per_year=periods,
    )
