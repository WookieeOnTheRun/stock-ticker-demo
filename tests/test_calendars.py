from datetime import date

import pytest

from stockapp.calendars import calendar_code, is_stale, sessions_per_year


@pytest.mark.parametrize(
    ("ticker", "code"),
    [("MSFT", "XNYS"), ("BRK-B", "XNYS"), ("VOD.L", "XLON"), ("7203.T", "XTKS"), ("^IRX", None), ("GBPUSD=X", None)],
)
def test_calendar_code(ticker, code):
    assert calendar_code(ticker) == code


def test_tel_aviv_is_not_mapped():
    assert calendar_code("TEVA.TA") is None


def test_sessions_per_year():
    assert sessions_per_year("MSFT", date(2026, 10, 16)) == pytest.approx(252, abs=3)
    assert sessions_per_year("7203.T", date(2026, 10, 16)) < 250
    assert sessions_per_year("^IRX", date(2026, 10, 16)) == 252


@pytest.mark.parametrize(
    ("last_bar", "today", "stale"),
    [
        (date(2026, 10, 19), date(2026, 10, 20), False),  # Mon bar, Tue: previous session present
        (date(2026, 10, 16), date(2026, 10, 18), False),  # Fri bar on Sunday
        (date(2026, 10, 16), date(2026, 10, 19), False),  # Fri bar on Monday
        (date(2026, 10, 16), date(2026, 10, 20), True),  # Fri bar on Tuesday: Monday is missing
    ],
)
def test_is_stale_with_calendar(last_bar, today, stale):
    assert is_stale("MSFT", last_bar, today) is stale


def test_is_stale_fallback_uses_four_calendar_days():
    assert not is_stale("^IRX", date(2026, 10, 16), date(2026, 10, 20))
    assert is_stale("^IRX", date(2026, 10, 15), date(2026, 10, 20))
