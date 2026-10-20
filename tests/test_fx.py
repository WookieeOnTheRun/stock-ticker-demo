import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockapp.fx import MINOR_UNITS, convert_prices, fx_ticker, major_currency

DATA = Path(__file__).parent / "data"
CURRENCIES = json.loads((DATA / "currencies.json").read_text())["currencies"]


def _series(name: str) -> pd.Series:
    """A recorded close series (yfinance 1.7.0, 2026-09/10)."""
    frame = pd.read_csv(DATA / name, index_col="Date", parse_dates=["Date"])
    return frame["close"]


def test_every_recorded_minor_unit_code_is_mapped():
    recorded = {CURRENCIES["VOD.L"], CURRENCIES["NPN.JO"], CURRENCIES["TEVA.TA"]}
    assert recorded == {"GBp", "ZAc", "ILA"}
    assert recorded <= set(MINOR_UNITS)
    assert major_currency("GBp") == ("GBP", 100.0)
    assert major_currency("ZAc") == ("ZAR", 100.0)
    assert major_currency("ILA") == ("ILS", 100.0)
    assert major_currency(CURRENCIES["MSFT"]) == ("USD", 1.0)


def test_fx_ticker_format():
    assert fx_ticker("GBP", "USD") == "GBPUSD=X"


@pytest.mark.parametrize(
    ("ticker", "native_file", "fx_file"),
    [
        ("VOD.L", "vod_l_native.csv", "vod_l_gbpusd.csv"),
        ("NPN.JO", "npn_jo_native.csv", "npn_jo_zarusd.csv"),
    ],
)
def test_recorded_minor_unit_prices_convert_to_usd(ticker, native_file, fx_file):
    native, rate = _series(native_file), _series(fx_file)
    major, divisor = major_currency(CURRENCIES[ticker])
    assert fx_ticker(major, "USD") == f"{fx_file[-10:-7].upper()}USD=X"

    usd = convert_prices(native, divisor, rate)

    assert usd.index.equals(native.index)
    common = native.index.intersection(rate.index)
    np.testing.assert_allclose(usd[common], native[common] / 100 * rate[common])


def test_minor_unit_without_fx_only_divides():
    native = _series("vod_l_native.csv")
    major, divisor = major_currency("GBp")
    np.testing.assert_allclose(convert_prices(native, divisor, None), native / 100)


def test_fx_is_forward_filled_never_back_filled():
    native = _series("vod_l_native.csv")
    rate = _series("vod_l_gbpusd.csv")
    rate = rate[rate.index >= native.index[2]]  # FX starts after the first two stock dates
    rate = rate.drop(native.index[5])  # and is missing one later stock date

    usd = convert_prices(native, 100.0, rate)

    assert native.index[0] not in usd.index and native.index[1] not in usd.index  # dropped, not back-filled
    filled_from = rate[rate.index < native.index[5]].iloc[-1]
    assert usd[native.index[5]] == pytest.approx(native[native.index[5]] / 100 * filled_from)
