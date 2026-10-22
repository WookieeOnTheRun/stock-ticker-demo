"""Currency handling: minor-unit quote currencies, FX pair symbols and conversion of prices to a base currency."""

import pandas as pd

from stockapp.riskfree import align_rf

# Minor-unit currency codes -> (major code, units per major unit). Only codes actually observed from yfinance 1.7.0
# (Ticker.history_metadata["currency"], checked 2026-10-20) are listed:
#   GBp: VOD.L, BP.L (London, pence)
#   ZAc: NPN.JO (Johannesburg, cents)
#   ILA: TEVA.TA, LUMI.TA (Tel Aviv, agorot)
# FX pairs must use the major code: "GBpUSD=X" returns the GBPUSD rate, not a hundredth of it.
MINOR_UNITS: dict[str, tuple[str, float]] = {
    "GBp": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0),
    "ILA": ("ILS", 100.0),
}

BASE_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "HKD"]


def major_currency(code: str) -> tuple[str, float]:
    """(major currency code, divisor) for a quote currency; divisor is 100 for verified minor-unit codes, else 1."""
    return MINOR_UNITS.get(code, (code, 1.0))


def fx_ticker(currency: str, base: str) -> str:
    """yfinance symbol for the rate that converts one unit of `currency` into `base`, e.g. GBPUSD=X."""
    return f"{currency}{base}=X"


def convert_prices(prices: pd.Series, divisor: float, fx_rate: pd.Series | None) -> pd.Series:
    """
    Convert native-currency prices to the base currency: prices / divisor * fx_rate.

    fx_rate (base units per major unit) is aligned to the price dates by forward-filling at most 5 days and never
    back-filling; price dates before the first FX observation, or with a stale rate, are dropped. Pass
    fx_rate=None when the major currency already is the base currency.
    """
    major = prices / divisor
    if fx_rate is None:
        return major
    rate = align_rf(fx_rate, pd.DatetimeIndex(prices.index))
    converted: pd.Series = major.loc[rate.index] * rate
    converted.name = prices.name
    return converted
