"""Watchlist input: typed tickers and uploaded CSVs, validated, and holdings converted to weights."""

import io
import re
from dataclasses import dataclass

import pandas as pd

from stockapp.data import normalize_ticker

MAX_UPLOAD_BYTES = 1024 * 1024  # 1 MB
MAX_TICKERS = 50
ALLOWED_COLUMNS = ("ticker", "weight", "shares")
COLUMN_DTYPES = {"ticker": "string", "weight": "float64", "shares": "float64"}
WEIGHT_TOLERANCE = 1e-6


class WatchlistError(ValueError):
    """The watchlist input is invalid; the message is safe to show to the user."""


@dataclass(frozen=True)
class Watchlist:
    """Validated tickers and, optionally, current holdings as weights or as share counts."""

    tickers: list[str]
    weights: dict[str, float] | None = None
    """Current holdings as portfolio weights (fractions; the rest is cash)."""
    shares: dict[str, float] | None = None
    """Current holdings as share counts (converted to weights with the latest prices)."""


def _validate_tickers(raw: list[str]) -> list[str]:
    """Normalized, de-duplicated tickers in input order; WatchlistError names up to 5 invalid ones."""
    tickers, invalid = [], []
    for item in raw:
        ticker = normalize_ticker(item)
        if ticker is None:
            invalid.append(item)
        elif ticker not in tickers:
            tickers.append(ticker)
    if invalid:
        shown = ", ".join(repr(t[:20]) for t in invalid[:5])
        raise WatchlistError(f"Invalid ticker(s): {shown}. Use letters, digits, and . ^ = - only (max 15 characters).")
    if len(tickers) > MAX_TICKERS:
        raise WatchlistError(f"At most {MAX_TICKERS} tickers are allowed; got {len(tickers)}.")
    return tickers


def parse_ticker_text(text: str) -> list[str]:
    """Tickers from free text separated by commas, spaces or new lines; validated, de-duplicated, order kept."""
    return _validate_tickers([t for t in re.split(r"[\s,;]+", text) if t])


def parse_watchlist_csv(data: bytes) -> Watchlist:
    """
    Parse an uploaded watchlist CSV (at most 1 MB). Allowed columns: `ticker` (required) plus optionally `weight` or
    `shares`; any other column is rejected. At most MAX_TICKERS rows; every ticker is validated with
    normalize_ticker. Parsed with pandas.read_csv using explicit usecols and dtypes.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise WatchlistError(f"The file is larger than {MAX_UPLOAD_BYTES // 1024} KB.")
    # Read the header alone first, so the columns are checked before any data is parsed with typed dtypes.
    try:
        header = pd.read_csv(io.BytesIO(data), nrows=0)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise WatchlistError("The file is not a readable CSV.") from exc
    columns = [str(c).strip().lower() for c in header.columns]
    extra = [c for c in columns if c not in ALLOWED_COLUMNS]
    if extra:
        raise WatchlistError(f"Unexpected column(s): {', '.join(extra[:5])}. Allowed: ticker, weight, shares.")
    if "ticker" not in columns:
        raise WatchlistError("The CSV needs a 'ticker' column.")
    if "weight" in columns and "shares" in columns:
        raise WatchlistError("Give holdings as either 'weight' or 'shares', not both.")
    if len(columns) != len(set(columns)):
        raise WatchlistError("Duplicate column names.")
    try:
        frame = pd.read_csv(
            io.BytesIO(data),
            usecols=lambda c: str(c).strip().lower() in columns,
            dtype={c: COLUMN_DTYPES[str(c).strip().lower()] for c in header.columns},
            nrows=MAX_TICKERS + 1,  # one extra row reveals an over-long file without parsing all of it
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise WatchlistError("Could not read the CSV: weight and shares must be numbers.") from exc
    frame.columns = columns
    if len(frame) > MAX_TICKERS:
        raise WatchlistError(f"At most {MAX_TICKERS} tickers are allowed.")
    frame = frame.dropna(subset=["ticker"])
    tickers = _validate_tickers([str(t) for t in frame["ticker"]])
    normalized = [normalize_ticker(str(t)) for t in frame["ticker"]]
    holdings: dict[str, float] = {}
    # At most one of weight / shares is present (checked above); duplicate tickers' holdings are summed.
    for column in ("weight", "shares"):
        if column in frame:
            values = frame[column].fillna(0.0)
            if (values < 0).any():
                raise WatchlistError(f"Negative {column} values are not allowed (no shorting).")
            for ticker, value in zip(normalized, values, strict=True):
                if ticker is not None:
                    holdings[ticker] = holdings.get(ticker, 0.0) + float(value)
            if column == "weight":
                return Watchlist(tickers, weights=validate_weights(holdings))
            return Watchlist(tickers, shares=holdings)
    return Watchlist(tickers)


def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    """Holdings weights must be non-negative and sum to at most 1 (the rest is cash)."""
    if any(w < 0 for w in weights.values()):
        raise WatchlistError("Weights must be non-negative.")
    total = sum(weights.values())
    if total > 1 + WEIGHT_TOLERANCE:
        raise WatchlistError(f"Weights sum to {total:.4f}; they must sum to at most 1 (the rest is cash).")
    return weights


def shares_to_weights(shares: dict[str, float], latest_prices: dict[str, float]) -> tuple[dict[str, float], float]:
    """Weights from share counts and latest prices (base currency), and the total holdings value."""
    values = {t: shares[t] * latest_prices[t] for t in shares if t in latest_prices}
    total = sum(values.values())
    if total <= 0:
        return {t: 0.0 for t in shares}, 0.0
    return {t: v / total for t, v in values.items()}, total
