import pandas as pd
import pytest

from stockapp.riskfree import FFILL_LIMIT, align_rf, daily_rf


def test_daily_rf_compounds_back_to_annual():
    d = daily_rf(pd.Series([5.0]))
    assert (1 + d.iloc[0]) ** 252 == pytest.approx(1.05, abs=1e-9)


def test_align_rf_forward_fills_only_and_drops_dates_before_first_rf():
    rf = pd.Series([0.01, 0.03], index=pd.to_datetime(["2024-01-03", "2024-01-05"]))
    stock_dates = pd.date_range("2024-01-02", "2024-01-08", name="Date")

    aligned = align_rf(rf, stock_dates)

    assert pd.Timestamp("2024-01-02") not in aligned.index  # before the first rf value: dropped, not back-filled
    assert aligned[pd.Timestamp("2024-01-03")] == 0.01
    assert aligned[pd.Timestamp("2024-01-04")] == 0.01  # carried forward from Jan 3, not back from Jan 5
    assert aligned[pd.Timestamp("2024-01-05")] == 0.03
    assert aligned[pd.Timestamp("2024-01-08")] == 0.03


def test_align_rf_drops_dates_beyond_fill_limit():
    rf = pd.Series([0.01], index=pd.to_datetime(["2024-01-01"]))
    stock_dates = pd.date_range("2024-01-01", periods=FFILL_LIMIT + 3)

    aligned = align_rf(rf, stock_dates)

    assert len(aligned) == FFILL_LIMIT + 1
