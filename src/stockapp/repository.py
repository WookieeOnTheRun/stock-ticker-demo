import logging
import time
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from stockapp import config
from stockapp.data import normalize_ticker
from stockapp.providers.base import PriceProvider
from stockapp.providers.errors import ProviderUnavailable, RateLimited
from stockapp.providers.yfinance_provider import YFinanceProvider
from stockapp.store import FetchLog, Store

logger = logging.getLogger(__name__)

OVERLAP_BARS = 10  # cached trading days re-fetched on each incremental update to detect restatements
RESTATEMENT_TOLERANCE = 1e-6  # max relative adj_close difference on overlapping dates before a full refetch


def get_provider() -> PriceProvider:
    """The configured price provider (yfinance; no keyed provider is configured)."""
    return YFinanceProvider()


def history_start(today: date, years: int | None) -> date | None:
    """First date of a `years`-long history ending today; None means the full available history."""
    if years is None:
        return None
    return (pd.Timestamp(today) - pd.DateOffset(years=years)).date()


def _covers(log: FetchLog, start: date | None) -> bool:
    """True if the cached history was fetched from `start` or earlier."""
    if log.history_start is None:
        return True
    return start is not None and log.history_start <= start


def _full_refresh(provider: PriceProvider, store: Store, ticker: str, start: date | None, now: datetime) -> None:
    """Fetch the full history from `start` and replace the ticker's cached rows in one transaction."""
    bars = provider.get_daily_bars(ticker, start, None)
    log = FetchLog(
        last_fetched_utc=now,
        first_date=bars.index[0].date(),
        last_date=bars.index[-1].date(),
        currency=provider.get_currency(ticker),
        history_start=start,
    )
    store.upsert_bars(
        provider.name, ticker, bars, actions=provider.get_corporate_actions(ticker), log=log, replace=True
    )


def _incremental_update(provider: PriceProvider, store: Store, ticker: str, log: FetchLog, now: datetime) -> None:
    """
    Fetch from OVERLAP_BARS cached trading days before the last cached date to today.

    Restatement check: if adj_close on any overlapping date differs from the cache by more than
    RESTATEMENT_TOLERANCE (relative), a split or dividend has re-based the adjusted series, so the whole
    history is refetched and replaced. The newest cached bar is left out of the comparison because it may
    have been stored mid-session, before the close was final.
    """
    cached = store.get_bars(provider.name, ticker)
    overlap_start = cached.index[max(len(cached) - OVERLAP_BARS - 1, 0)].date()
    fresh = provider.get_daily_bars(ticker, overlap_start, None)

    comparable = cached.index[:-1].intersection(fresh.index)
    old = cached.loc[comparable, "adj_close"]
    new = fresh.loc[comparable, "adj_close"]
    relative_diff = ((new - old).abs() / old.abs()).max() if len(comparable) else float("inf")
    if not relative_diff <= RESTATEMENT_TOLERANCE:
        logger.info(
            "Restatement detected for %s (max relative adj_close change %.3g on %d overlapping days); "
            "refetching full history",
            ticker,
            relative_diff,
            len(comparable),
            extra={"event": "restatement_refetch", "ticker": ticker, "provider": provider.name},
        )
        _full_refresh(provider, store, ticker, log.history_start, now)
        return

    new_log = FetchLog(
        last_fetched_utc=now,
        first_date=min(log.first_date, fresh.index[0].date()),
        last_date=fresh.index[-1].date(),
        currency=provider.get_currency(ticker),
        history_start=log.history_start,
    )
    store.upsert_bars(provider.name, ticker, fresh, actions=provider.get_corporate_actions(ticker), log=new_log)


def load_bars(
    ticker: str,
    years: int | None = 10,
    *,
    provider: PriceProvider | None = None,
    store: Store | None = None,
    now: datetime | None = None,
    freshness: timedelta | None = None,
) -> pd.DataFrame:
    """
    Daily bars (Bars schema) for the last `years` years (None = all available), served from the SQLite cache.

    1. Cold (or a longer history than cached): fetch the full history and store it.
    2. Cache fetched within `freshness` (default config.FRESHNESS_HOURS): return it without a request.
    3. Otherwise fetch only the missing bars plus an overlap, checking the overlap for restatements.
    If the provider is unavailable or rate limited and a cache exists, the cache is returned as is.
    """
    normalized = normalize_ticker(ticker)
    if normalized is None:
        raise ValueError(f"Invalid ticker: {ticker!r}")
    provider = provider if provider is not None else get_provider()
    store = store if store is not None else Store()
    now = now if now is not None else datetime.now(UTC)
    freshness = freshness if freshness is not None else timedelta(hours=config.FRESHNESS_HOURS)
    start = history_start(now.date(), years)

    started = time.perf_counter()
    log = store.get_fetch_log(provider.name, normalized)
    cache = "hit"
    try:
        if log is None or not _covers(log, start):
            cache = "miss"
            _full_refresh(provider, store, normalized, start, now)
        elif now - log.last_fetched_utc >= freshness:
            cache = "incremental"
            _incremental_update(provider, store, normalized, log, now)
    except (ProviderUnavailable, RateLimited) as exc:
        if log is None:
            raise
        cache = "stale_fallback"
        logger.warning("Serving cached bars for %s: %s", normalized, exc)
    bars = store.get_bars(provider.name, normalized, start)
    logger.info(
        "data fetch",
        extra={
            "event": "data_fetch",
            "ticker": normalized,
            "provider": provider.name,
            "cache": cache,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return bars


def load_currency(ticker: str, *, provider: PriceProvider | None = None, store: Store | None = None) -> str | None:
    """Quote currency recorded for a cached ticker (call load_bars first), or None if unknown."""
    provider_name = provider.name if provider is not None else YFinanceProvider.name
    log = (store if store is not None else Store()).get_fetch_log(provider_name, ticker)
    return log.currency if log else None


def load_actions(ticker: str, *, provider: PriceProvider | None = None, store: Store | None = None) -> pd.DataFrame:
    """Cached splits and dividends for a ticker (call load_bars first)."""
    provider_name = provider.name if provider is not None else YFinanceProvider.name
    return (store if store is not None else Store()).get_actions(provider_name, ticker)
