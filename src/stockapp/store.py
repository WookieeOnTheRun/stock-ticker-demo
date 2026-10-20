import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from stockapp import config
from stockapp.data import normalize_ticker
from stockapp.providers.base import ACTION_COLUMNS, BAR_COLUMNS, Bars

PROVIDER_PATTERN = re.compile(r"[a-z0-9_]{1,32}")

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    provider TEXT NOT NULL, ticker TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
    PRIMARY KEY (provider, ticker, date)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    provider TEXT NOT NULL, ticker TEXT NOT NULL,
    last_fetched_utc TEXT NOT NULL, first_date TEXT NOT NULL, last_date TEXT NOT NULL,
    currency TEXT, history_start TEXT,
    PRIMARY KEY (provider, ticker)
);
CREATE TABLE IF NOT EXISTS actions (
    provider TEXT NOT NULL, ticker TEXT NOT NULL, date TEXT NOT NULL, type TEXT NOT NULL, value REAL,
    PRIMARY KEY (provider, ticker, date, type)
);
CREATE TABLE IF NOT EXISTS factors (
    date TEXT PRIMARY KEY, mkt_rf REAL, smb REAL, hml REAL, rmw REAL, cma REAL, rf REAL, mom REAL
);
CREATE TABLE IF NOT EXISTS factor_log (
    id INTEGER PRIMARY KEY CHECK (id = 1), last_fetched_utc TEXT NOT NULL
);
"""

FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"]


@dataclass(frozen=True)
class FetchLog:
    last_fetched_utc: datetime
    first_date: date
    last_date: date
    currency: str | None
    history_start: date | None
    """Earliest start date requested so far; None means the full available history was requested."""


def resolve_db_path(path: Path) -> Path:
    """Resolve path (relative paths against the project root) and reject anything outside config.DATA_DIR."""
    resolved = (path if path.is_absolute() else config.PROJECT_ROOT / path).resolve()
    data_dir = config.DATA_DIR.resolve()
    if not resolved.is_relative_to(data_dir):
        raise ValueError(f"Database path must be inside {data_dir}")
    return resolved


def _check_keys(provider: str, ticker: str) -> None:
    """Validate identifiers before any I/O; tickers must already be in normalize_ticker's canonical form."""
    if not PROVIDER_PATTERN.fullmatch(provider):
        raise ValueError(f"Invalid provider name: {provider!r}")
    if normalize_ticker(ticker) != ticker:
        raise ValueError(f"Invalid ticker: {ticker!r}")


class Store:
    """
    SQLite cache for bars, corporate actions and a per-ticker fetch log. Opens one connection per call
    (Streamlit runs scripts on several threads, so connections are never shared), uses WAL mode, and
    writes each update in a single transaction with parameterized queries only.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = resolve_db_path(path if path is not None else config.DB_PATH)

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

    def get_bars(self, provider: str, ticker: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Cached bars for start..end inclusive (None = unbounded), in the Bars schema."""
        _check_keys(provider, ticker)
        query = f"SELECT date, {', '.join(BAR_COLUMNS)} FROM bars WHERE provider = ? AND ticker = ?"  # noqa: S608 - fixed column list
        params: list[str] = [provider, ticker]
        if start is not None:
            query += " AND date >= ?"
            params.append(start.isoformat())
        if end is not None:
            query += " AND date <= ?"
            params.append(end.isoformat())
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY date", params).fetchall()
        if not rows:
            return Bars.empty()
        frame = pd.DataFrame(rows, columns=["date", *BAR_COLUMNS])
        index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date")), name="Date")
        return Bars.validate(frame.set_index(index).astype("float64"))

    def get_fetch_log(self, provider: str, ticker: str) -> FetchLog | None:
        _check_keys(provider, ticker)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_fetched_utc, first_date, last_date, currency, history_start "
                "FROM fetch_log WHERE provider = ? AND ticker = ?",
                (provider, ticker),
            ).fetchone()
        if row is None:
            return None
        return FetchLog(
            last_fetched_utc=datetime.fromisoformat(row[0]),
            first_date=date.fromisoformat(row[1]),
            last_date=date.fromisoformat(row[2]),
            currency=row[3],
            history_start=date.fromisoformat(row[4]) if row[4] else None,
        )

    def get_actions(self, provider: str, ticker: str) -> pd.DataFrame:
        """Cached splits and dividends: columns date, type, value."""
        _check_keys(provider, ticker)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, type, value FROM actions WHERE provider = ? AND ticker = ? ORDER BY date, type",
                (provider, ticker),
            ).fetchall()
        actions = pd.DataFrame(rows, columns=ACTION_COLUMNS)
        actions["date"] = pd.to_datetime(actions["date"])
        actions["value"] = actions["value"].astype("float64")
        return actions

    def upsert_bars(
        self,
        provider: str,
        ticker: str,
        bars: pd.DataFrame,
        *,
        actions: pd.DataFrame | None = None,
        log: FetchLog | None = None,
        replace: bool = False,
    ) -> None:
        """
        Insert or update bars (and optionally actions and the fetch log) in one transaction.
        With replace=True the ticker's existing bars and actions are deleted first, in the same transaction.
        """
        _check_keys(provider, ticker)
        Bars.validate(bars)
        bar_rows = [
            (provider, ticker, ts.date().isoformat(), *(None if pd.isna(v) else float(v) for v in values))
            for ts, values in zip(bars.index, bars[BAR_COLUMNS].itertuples(index=False, name=None), strict=True)
        ]
        with self._connect() as conn, conn:
            if replace:
                conn.execute("DELETE FROM bars WHERE provider = ? AND ticker = ?", (provider, ticker))
                conn.execute("DELETE FROM actions WHERE provider = ? AND ticker = ?", (provider, ticker))
            conn.executemany(
                "INSERT INTO bars (provider, ticker, date, open, high, low, close, adj_close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (provider, ticker, date) DO UPDATE SET open = excluded.open, high = excluded.high, "
                "low = excluded.low, close = excluded.close, adj_close = excluded.adj_close, "
                "volume = excluded.volume",
                bar_rows,
            )
            if actions is not None and not actions.empty:
                conn.executemany(
                    "INSERT INTO actions (provider, ticker, date, type, value) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (provider, ticker, date, type) DO UPDATE SET value = excluded.value",
                    [
                        (provider, ticker, pd.Timestamp(d).date().isoformat(), str(t), float(v))
                        for d, t, v in actions[ACTION_COLUMNS].itertuples(index=False, name=None)
                    ],
                )
            if log is not None:
                conn.execute(
                    "INSERT INTO fetch_log (provider, ticker, last_fetched_utc, first_date, last_date, currency, "
                    "history_start) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (provider, ticker) DO UPDATE SET last_fetched_utc = excluded.last_fetched_utc, "
                    "first_date = excluded.first_date, last_date = excluded.last_date, "
                    "currency = excluded.currency, history_start = excluded.history_start",
                    (
                        provider,
                        ticker,
                        log.last_fetched_utc.isoformat(),
                        log.first_date.isoformat(),
                        log.last_date.isoformat(),
                        log.currency,
                        log.history_start.isoformat() if log.history_start else None,
                    ),
                )

    def delete_ticker(self, provider: str, ticker: str) -> None:
        """Remove a ticker's bars, actions and fetch log in one transaction."""
        _check_keys(provider, ticker)
        with self._connect() as conn, conn:
            for table in ("bars", "actions", "fetch_log"):
                conn.execute(f"DELETE FROM {table} WHERE provider = ? AND ticker = ?", (provider, ticker))  # noqa: S608 - fixed table names

    def get_factors(self) -> tuple[pd.DataFrame, datetime | None]:
        """Cached daily factors (decimal returns, DatetimeIndex "Date") and when they were last fetched."""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT date, {', '.join(FACTOR_COLUMNS)} FROM factors ORDER BY date").fetchall()  # noqa: S608 - fixed column list
            log = conn.execute("SELECT last_fetched_utc FROM factor_log WHERE id = 1").fetchone()
        frame = pd.DataFrame(rows, columns=["date", *FACTOR_COLUMNS])
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date")), name="Date")
        return frame.astype("float64"), datetime.fromisoformat(log[0]) if log else None

    def replace_factors(self, factors: pd.DataFrame, fetched_utc: datetime) -> None:
        """Replace the cached factor table and its fetch time in one transaction."""
        rows = [
            (ts.date().isoformat(), *(float(v) for v in values))
            for ts, values in zip(
                factors.index, factors[FACTOR_COLUMNS].itertuples(index=False, name=None), strict=True
            )
        ]
        with self._connect() as conn, conn:
            conn.execute("DELETE FROM factors")
            conn.executemany(
                f"INSERT INTO factors (date, {', '.join(FACTOR_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",  # noqa: S608 - fixed column list
                rows,
            )
            conn.execute(
                "INSERT INTO factor_log (id, last_fetched_utc) VALUES (1, ?) "
                "ON CONFLICT (id) DO UPDATE SET last_fetched_utc = excluded.last_fetched_utc",
                (fetched_utc.isoformat(),),
            )
