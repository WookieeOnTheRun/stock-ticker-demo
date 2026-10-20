"""
Per-user saved watchlists and settings in SQLite (data/users.sqlite). Users are keyed only by sub_hash, the
sha256(app_salt + sub) digest; no raw identity claim is ever stored. Every JSON payload is validated against an
explicit schema on write and again on read, and anything else is rejected.
"""

import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from stockapp import config
from stockapp.backtest.strategies import KELLY_METHODS
from stockapp.data import normalize_ticker
from stockapp.fx import BASE_CURRENCIES
from stockapp.store import resolve_db_path

USERS_DB_FILE = "users.sqlite"
MAX_WATCHLIST_TICKERS = 50
MAX_WATCHLISTS = 100
SUB_HASH = re.compile(r"[0-9a-f]{64}")
WATCHLIST_NAME = re.compile(r"[A-Za-z0-9 _.\-]{1,40}")

HISTORY_LABELS = ("5y", "10y", "20y", "max")
BENCHMARK_CHOICES = ("SPY", "QQQ", "IWM", "DIA", "Other")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (sub_hash TEXT PRIMARY KEY, created_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS watchlists (
    sub_hash TEXT NOT NULL, name TEXT NOT NULL, tickers_json TEXT NOT NULL, updated_utc TEXT NOT NULL,
    PRIMARY KEY (sub_hash, name)
);
CREATE TABLE IF NOT EXISTS settings (sub_hash TEXT PRIMARY KEY, settings_json TEXT NOT NULL, updated_utc TEXT NOT NULL);
"""


class UserDataError(ValueError):
    """User data failed validation; the message is safe to show."""


def _ticker(value: object) -> bool:
    return isinstance(value, str) and normalize_ticker(value) == value


Validator = Callable[[object], bool]


def _number(lo: float, hi: float, integer: bool = False) -> Validator:
    def check(value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        if integer and not float(value).is_integer():
            return False
        return lo <= float(value) <= hi

    return check


def _choice(options: tuple[str, ...]) -> Validator:
    return lambda value: isinstance(value, str) and value in options


SETTINGS_SCHEMA: dict[str, Validator] = {
    "ticker": _ticker,
    "history": _choice(HISTORY_LABELS),
    "base_currency": _choice(tuple(BASE_CURRENCIES)),
    "benchmark_choice": _choice(BENCHMARK_CHOICES),
    "benchmark_other": _ticker,
    "k_fraction": _number(0.1, 1.0),
    "method": _choice(KELLY_METHODS),
    "prior_sd_annual": _number(0.01, 0.50),
    "max_dd_pct": _number(10, 50, integer=True),
    "lookback": _number(60, 20_000, integer=True),
}
"""Whitelisted setting keys and their validators (types and ranges match the sidebar widgets)."""


def validate_sub_hash(sub_hash: str) -> str:
    if not isinstance(sub_hash, str) or not SUB_HASH.fullmatch(sub_hash):
        raise UserDataError("Invalid user key.")
    return sub_hash


def validate_tickers(tickers: object) -> list[str]:
    """A list of at most 50 distinct tickers, each already in normalize_ticker's canonical form."""
    if not isinstance(tickers, list):
        raise UserDataError("A watchlist must be a list of tickers.")
    if len(tickers) > MAX_WATCHLIST_TICKERS:
        raise UserDataError(f"A watchlist can hold at most {MAX_WATCHLIST_TICKERS} tickers.")
    if not all(_ticker(t) for t in tickers):
        raise UserDataError("A watchlist contains an invalid ticker.")
    if len(set(tickers)) != len(tickers):
        raise UserDataError("A watchlist contains duplicate tickers.")
    return list(tickers)


def validate_settings(settings: object) -> dict[str, object]:
    """Only whitelisted keys, each passing its typed range check."""
    if not isinstance(settings, dict):
        raise UserDataError("Settings must be a mapping.")
    unknown = sorted(set(settings) - set(SETTINGS_SCHEMA))
    if unknown:
        raise UserDataError(f"Unknown setting(s): {', '.join(str(k)[:30] for k in unknown[:5])}.")
    bad = [k for k, v in settings.items() if not SETTINGS_SCHEMA[k](v)]
    if bad:
        raise UserDataError(f"Invalid value for setting(s): {', '.join(bad)}.")
    return dict(settings)


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not WATCHLIST_NAME.fullmatch(name):
        raise UserDataError("Watchlist names are 1-40 letters, digits, spaces, '.', '_' or '-'.")
    return name


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class UserStore:
    """SQLite store for per-user data: WAL mode, one connection per call, parameterized queries only."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = resolve_db_path(path if path is not None else config.DATA_DIR / USERS_DB_FILE)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _ensure_user(conn: sqlite3.Connection, sub_hash: str) -> None:
        conn.execute("INSERT OR IGNORE INTO users (sub_hash, created_utc) VALUES (?, ?)", (sub_hash, _now()))

    def save_watchlist(self, sub_hash: str, name: str, tickers: list[str]) -> None:
        validate_sub_hash(sub_hash)
        validate_name(name)
        payload = json.dumps(validate_tickers(tickers))
        with self._connect() as conn, conn:
            count = conn.execute("SELECT COUNT(*) FROM watchlists WHERE sub_hash = ?", (sub_hash,)).fetchone()[0]
            exists = conn.execute(
                "SELECT 1 FROM watchlists WHERE sub_hash = ? AND name = ?", (sub_hash, name)
            ).fetchone()
            if count >= MAX_WATCHLISTS and not exists:
                raise UserDataError(f"At most {MAX_WATCHLISTS} saved watchlists.")
            self._ensure_user(conn, sub_hash)
            conn.execute(
                "INSERT INTO watchlists (sub_hash, name, tickers_json, updated_utc) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (sub_hash, name) DO UPDATE SET tickers_json = excluded.tickers_json, "
                "updated_utc = excluded.updated_utc",
                (sub_hash, name, payload, _now()),
            )

    def list_watchlists(self, sub_hash: str) -> list[str]:
        validate_sub_hash(sub_hash)
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM watchlists WHERE sub_hash = ? ORDER BY name", (sub_hash,)).fetchall()
        return [str(r[0]) for r in rows]

    def load_watchlist(self, sub_hash: str, name: str) -> list[str]:
        """The saved tickers, validated again on read; raises UserDataError if missing or invalid."""
        validate_sub_hash(sub_hash)
        validate_name(name)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT tickers_json FROM watchlists WHERE sub_hash = ? AND name = ?", (sub_hash, name)
            ).fetchone()
        if row is None:
            raise UserDataError("No such watchlist.")
        try:
            return validate_tickers(json.loads(row[0]))
        except json.JSONDecodeError as exc:
            raise UserDataError("Saved watchlist is corrupt.") from exc

    def delete_watchlist(self, sub_hash: str, name: str) -> None:
        validate_sub_hash(sub_hash)
        with self._connect() as conn, conn:
            conn.execute("DELETE FROM watchlists WHERE sub_hash = ? AND name = ?", (sub_hash, name))

    def save_settings(self, sub_hash: str, settings: dict[str, object]) -> None:
        validate_sub_hash(sub_hash)
        payload = json.dumps(validate_settings(settings), sort_keys=True)
        with self._connect() as conn, conn:
            self._ensure_user(conn, sub_hash)
            conn.execute(
                "INSERT INTO settings (sub_hash, settings_json, updated_utc) VALUES (?, ?, ?) "
                "ON CONFLICT (sub_hash) DO UPDATE SET settings_json = excluded.settings_json, "
                "updated_utc = excluded.updated_utc",
                (sub_hash, payload, _now()),
            )

    def load_settings(self, sub_hash: str) -> dict[str, object] | None:
        """Saved settings validated on read, or None if none are saved; raises UserDataError if invalid."""
        validate_sub_hash(sub_hash)
        with self._connect() as conn:
            row = conn.execute("SELECT settings_json FROM settings WHERE sub_hash = ?", (sub_hash,)).fetchone()
        if row is None:
            return None
        try:
            return validate_settings(json.loads(row[0]))
        except json.JSONDecodeError as exc:
            raise UserDataError("Saved settings are corrupt.") from exc

    def reset_settings(self, sub_hash: str) -> None:
        validate_sub_hash(sub_hash)
        with self._connect() as conn, conn:
            conn.execute("DELETE FROM settings WHERE sub_hash = ?", (sub_hash,))

    def delete_user_data(self, sub_hash: str) -> None:
        """Remove every row for this user from every table, in one transaction."""
        validate_sub_hash(sub_hash)
        with self._connect() as conn, conn:
            for table in ("watchlists", "settings", "users"):
                conn.execute(f"DELETE FROM {table} WHERE sub_hash = ?", (sub_hash,))  # noqa: S608 - fixed table names
