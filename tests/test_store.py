from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fake_provider import FakeProvider

from stockapp import config
from stockapp.store import FetchLog, Store


def _log(bars):
    return FetchLog(datetime(2026, 10, 17, tzinfo=UTC), bars.index[0].date(), bars.index[-1].date(), "USD", None)


def test_round_trip_bars_actions_and_log():
    fake = FakeProvider(n=50)
    store = Store()
    store.upsert_bars("fake", "MSFT", fake.bars, actions=fake.actions, log=_log(fake.bars))

    pd.testing.assert_frame_equal(store.get_bars("fake", "MSFT"), fake.bars, check_freq=False)
    actions = store.get_actions("fake", "MSFT")
    assert actions["type"].tolist() == ["split", "dividend"]
    assert actions["value"].tolist() == [2.0, 0.5]
    assert store.get_fetch_log("fake", "MSFT") == _log(fake.bars)


def test_get_bars_respects_date_range():
    fake = FakeProvider(n=50)
    store = Store()
    store.upsert_bars("fake", "MSFT", fake.bars)
    start, end = fake.bars.index[10].date(), fake.bars.index[19].date()
    assert len(store.get_bars("fake", "MSFT", start, end)) == 10


def test_upsert_is_idempotent():
    fake = FakeProvider(n=50)
    store = Store()
    for _ in range(3):
        store.upsert_bars("fake", "MSFT", fake.bars, actions=fake.actions)
    assert len(store.get_bars("fake", "MSFT")) == 50
    assert len(store.get_actions("fake", "MSFT")) == 2


def test_upsert_overwrites_existing_dates():
    fake = FakeProvider(n=50)
    store = Store()
    store.upsert_bars("fake", "MSFT", fake.bars)
    changed = fake.bars.copy()
    changed["adj_close"] *= 2
    store.upsert_bars("fake", "MSFT", changed.tail(5))
    stored = store.get_bars("fake", "MSFT")
    assert (stored["adj_close"].tail(5) == changed["adj_close"].tail(5)).all()
    assert (stored["adj_close"].head(45) == fake.bars["adj_close"].head(45)).all()


@pytest.mark.parametrize("ticker", ["MSFT'; DROP TABLE bars;--", "msft", "", "A" * 16, "../etc"])
def test_invalid_ticker_rejected_before_io(ticker):
    store = Store()
    with pytest.raises(ValueError, match="Invalid ticker"):
        store.get_bars("fake", ticker)
    with pytest.raises(ValueError, match="Invalid ticker"):
        store.upsert_bars("fake", ticker, FakeProvider(n=5).bars)
    assert not store.path.exists()  # rejected before the database was ever opened


@pytest.mark.parametrize("path", [Path("/var/lib/market.sqlite"), Path("data/../outside.sqlite")])
def test_path_outside_data_dir_rejected(path):
    with pytest.raises(ValueError, match="must be inside"):
        Store(path)


def test_default_path_is_inside_data_dir(isolated_data_dir):
    assert Store().path.is_relative_to(isolated_data_dir)
    assert Store().path == config.DB_PATH.resolve()
