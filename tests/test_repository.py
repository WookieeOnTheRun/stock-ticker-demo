import logging
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from fake_provider import FakeProvider

from stockapp.providers.errors import ProviderUnavailable
from stockapp.repository import OVERLAP_BARS, load_actions, load_bars, load_currency
from stockapp.store import Store

NOW = datetime(2026, 10, 17, 12, tzinfo=UTC)
LATER = NOW + timedelta(hours=7)  # past the default 6-hour freshness window


def _load(fake, now=NOW, years=10):
    return load_bars("MSFT", years, provider=fake, now=now)


def test_cold_load_fetches_full_history_and_stores_it():
    fake = FakeProvider()
    bars = _load(fake)
    assert fake.calls == [("MSFT", date(2016, 10, 17), None)]
    pd.testing.assert_frame_equal(bars, fake.bars, check_freq=False)
    assert load_currency("MSFT", provider=fake) == "USD"
    assert len(load_actions("MSFT", provider=fake)) == 2


def test_warm_hit_within_freshness_makes_no_request():
    fake = FakeProvider()
    _load(fake)
    bars = _load(fake, now=NOW + timedelta(hours=5))
    assert len(fake.calls) == 1
    pd.testing.assert_frame_equal(bars, fake.bars, check_freq=False)


def test_incremental_update_fetches_only_overlap_and_new_bars():
    fake = FakeProvider()
    _load(fake)
    cached_index = fake.bars.index
    fake.append_days(3)

    bars = _load(fake, now=LATER)

    assert fake.calls[-1] == ("MSFT", cached_index[-OVERLAP_BARS - 1].date(), None)
    assert len(fake.calls) == 2  # no full refetch
    pd.testing.assert_frame_equal(bars, fake.bars, check_freq=False)


def test_restatement_in_overlap_triggers_full_refetch(caplog):
    fake = FakeProvider()
    _load(fake)
    overlap_date = fake.bars.index[-5]
    fake.bars.loc[overlap_date, "adj_close"] *= 1.01  # a dividend or split re-based the adjusted series
    fake.bars.loc[fake.bars.index[0], "adj_close"] *= 1.01  # outside the overlap: only a full refetch picks it up

    with caplog.at_level(logging.INFO, logger="stockapp.repository"):
        bars = _load(fake, now=LATER)

    assert fake.calls[-1] == ("MSFT", date(2016, 10, 17), None)  # full history requested again
    assert "Restatement detected for MSFT" in caplog.text
    pd.testing.assert_frame_equal(bars, fake.bars, check_freq=False)


def test_change_in_last_cached_bar_is_not_a_restatement():
    # The newest cached bar may have been stored mid-session, so a changed close there is just the final value.
    fake = FakeProvider()
    _load(fake)
    fake.bars.loc[fake.bars.index[-1], "adj_close"] *= 1.01

    bars = _load(fake, now=LATER)

    assert len(fake.calls) == 2  # incremental only
    assert bars["adj_close"].iloc[-1] == fake.bars["adj_close"].iloc[-1]


def test_longer_history_request_refetches():
    fake = FakeProvider()
    _load(fake, years=5)
    _load(fake, now=NOW + timedelta(hours=1), years=10)
    assert fake.calls[-1] == ("MSFT", date(2016, 10, 17), None)
    _load(fake, now=NOW + timedelta(hours=2), years=5)  # covered by the 10-year fetch, still fresh
    assert len(fake.calls) == 2


def test_provider_outage_serves_cache():
    fake = FakeProvider()
    _load(fake)
    fake.fail = True
    bars = _load(fake, now=LATER)
    pd.testing.assert_frame_equal(bars, fake.bars, check_freq=False)


def test_provider_outage_without_cache_raises():
    fake = FakeProvider()
    fake.fail = True
    with pytest.raises(ProviderUnavailable):
        _load(fake)


def test_invalid_ticker_rejected_before_io():
    fake = FakeProvider()
    with pytest.raises(ValueError, match="Invalid ticker"):
        load_bars("MSFT; DROP TABLE", provider=fake, now=NOW)
    assert fake.calls == []
    assert not Store().path.exists()


def test_warm_load_of_ten_years_is_fast():
    import time

    fake = FakeProvider(n=2520)
    _load(fake)
    start = time.perf_counter()
    _load(fake, now=NOW + timedelta(hours=1))
    assert time.perf_counter() - start < 0.1
