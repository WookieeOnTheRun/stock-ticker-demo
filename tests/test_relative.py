import numpy as np
import pandas as pd
import pytest

from stockapp import relative, seasonality


def test_sector_mapping_only_for_known_sectors():
    assert relative.sector_etf("Technology") == "XLK"
    assert relative.sector_etf("Financial Services") == "XLF"
    assert relative.sector_etf("Crypto") is None
    assert relative.sector_etf(None) is None
    assert len(set(relative.SECTOR_ETFS.values())) == 11


def test_growth_and_relative_table():
    index = pd.bdate_range("2024-01-01", periods=300)
    stock = pd.Series(100 * 1.001 ** np.arange(300), index=index)
    bench = pd.Series(50.0, index=index)
    growth = relative.growth_of_one({"stock": stock, "bench": bench})
    assert growth.iloc[0].tolist() == [1.0, 1.0]
    table = relative.relative_table(stock, {"bench": bench})
    assert table.loc["1M", "stock"] == pytest.approx(1.001**21 - 1)
    assert table.loc["1M", "bench"] == 0
    assert table.loc["12M", "stock vs bench"] == pytest.approx(1.001**252 - 1)


def test_monthly_returns_and_averages():
    prices = pd.Series(
        [100.0, 110.0, 99.0, 108.9], index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2025-02-28"])
    )
    monthly = seasonality.monthly_returns(prices)
    assert monthly.loc[2024, "Feb"] == pytest.approx(0.10)
    assert monthly.loc[2024, "Mar"] == pytest.approx(-0.10)
    assert monthly.loc[2025, "Feb"] == pytest.approx(0.10)  # from the last available month-end (Mar 2024)
    average = seasonality.average_by_month(monthly)
    assert average.loc["Feb", "count"] == 2
    assert average.loc["Feb", "mean"] == pytest.approx(0.10)
