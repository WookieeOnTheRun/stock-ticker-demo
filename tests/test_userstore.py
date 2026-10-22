from stockapp.userstore import UserStore

OLD, NEW = "a" * 64, "b" * 64


def test_migrate_user_moves_every_row():
    store = UserStore()
    store.save_watchlist(OLD, "Core", ["MSFT"])
    store.save_watchlist(OLD, "Tech", ["AAPL", "NVDA"])
    store.save_settings(OLD, {"ticker": "SPY"})

    assert store.migrate_user(OLD, NEW) is True

    assert store.list_watchlists(NEW) == ["Core", "Tech"]
    assert store.load_watchlist(NEW, "Tech") == ["AAPL", "NVDA"]
    assert store.load_settings(NEW) == {"ticker": "SPY"}
    assert store.list_watchlists(OLD) == [] and store.load_settings(OLD) is None


def test_migrate_user_without_old_data_is_a_no_op():
    store = UserStore()
    assert store.migrate_user(OLD, NEW) is False
    assert store.list_watchlists(NEW) == []


def test_migrate_user_never_merges_into_existing_data():
    store = UserStore()
    store.save_watchlist(OLD, "Old", ["MSFT"])
    store.save_watchlist(NEW, "New", ["AAPL"])

    assert store.migrate_user(OLD, NEW) is False

    assert store.list_watchlists(OLD) == ["Old"]
    assert store.list_watchlists(NEW) == ["New"]


def test_migrate_user_to_itself_is_a_no_op():
    store = UserStore()
    store.save_watchlist(OLD, "Core", ["MSFT"])
    assert store.migrate_user(OLD, OLD) is False
    assert store.list_watchlists(OLD) == ["Core"]
