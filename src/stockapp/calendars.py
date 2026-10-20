from datetime import date, timedelta
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from stockapp.kelly import TRADING_DAYS

STALE_FALLBACK_DAYS = 4  # without a calendar, flag data whose last bar is more than this many calendar days old

# Ticker suffix -> exchange calendar. Each entry was checked by comparing two years of yfinance bars
# (2024-10 to 2026-10) with the calendar's sessions: every bar fell on a session and at most one session lacked
# a bar. Sample tickers: MSFT/AAPL/SPY (no suffix), VOD.L, SHOP.TO, SAP.DE, 0700.HK, 7203.T, NPN.JO, CBA.AX,
# NESN.SW. ".TA" is deliberately absent: yfinance has no bars for Tel Aviv's pre-2026 Sunday sessions, so
# calendar-based annualization would not match the bars.
SUFFIX_CALENDARS = {
    "": "XNYS",
    ".L": "XLON",
    ".TO": "XTSE",
    ".DE": "XETR",
    ".HK": "XHKG",
    ".T": "XTKS",
    ".JO": "XJSE",
    ".AX": "XASX",
    ".SW": "XSWX",
}


def calendar_code(ticker: str) -> str | None:
    """Exchange calendar code for a ticker, or None for indices (^...), FX/futures (=) and unverified suffixes."""
    if ticker.startswith("^") or "=" in ticker:
        return None
    suffix = ticker[ticker.rfind(".") :] if "." in ticker else ""
    return SUFFIX_CALENDARS.get(suffix)


@lru_cache(maxsize=16)
def _calendar(code: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(code)


def sessions_per_year(ticker: str, as_of: date) -> float:
    """Trading sessions in the year ending `as_of` on the ticker's exchange; TRADING_DAYS (252) without a calendar."""
    code = calendar_code(ticker)
    if code is None:
        return float(TRADING_DAYS)
    cal = _calendar(code)
    start = max(pd.Timestamp(as_of - timedelta(days=365)), cal.first_session)
    end = min(pd.Timestamp(as_of), cal.last_session)
    return float(len(cal.sessions_in_range(start, end)))


def is_stale(ticker: str, last_bar: date, today: date) -> bool:
    """
    True if the last bar is older than the exchange's previous session (the last session before `today`).
    Without a calendar: True if the last bar is more than STALE_FALLBACK_DAYS calendar days old.
    """
    code = calendar_code(ticker)
    if code is None:
        return (today - last_bar).days > STALE_FALLBACK_DAYS
    cal = _calendar(code)
    ts = min(pd.Timestamp(today), cal.last_session)
    previous = cal.previous_session(ts) if cal.is_session(ts) else cal.date_to_session(ts, direction="previous")
    return bool(pd.Timestamp(last_bar) < previous)
