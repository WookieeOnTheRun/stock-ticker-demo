import pandas as pd

# yfinance "sector" strings -> SPDR sector ETFs. Every key was observed on a real ticker (yfinance 1.7.0, October
# 2026): AAPL Technology, JPM Financial Services, JNJ Healthcare, XOM Energy, CAT Industrials, AMZN Consumer
# Cyclical, PG Consumer Defensive, NEE Utilities, LIN Basic Materials, PLD Real Estate, GOOGL Communication Services.
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

RELATIVE_WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}


def sector_etf(sector: str | None) -> str | None:
    """The SPDR sector ETF for a yfinance sector string, or None if the sector is unknown or missing."""
    return SECTOR_ETFS.get(sector) if sector else None


def growth_of_one(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Growth of $1 for each series from their first common date: P_t / P_start, on common dates only."""
    joined = pd.concat(prices, axis=1, join="inner").dropna()
    return joined / joined.iloc[0]


def trailing_returns(prices: pd.Series, windows: dict[str, int] = RELATIVE_WINDOWS) -> pd.Series:
    """Simple return over the last n trading days for each window: P_t / P_{t-n} - 1 (NaN if history is shorter)."""
    return pd.Series(
        {
            name: prices.iloc[-1] / prices.iloc[-n - 1] - 1 if len(prices) > n else float("nan")
            for name, n in windows.items()
        }
    )


def relative_table(
    stock: pd.Series, others: dict[str, pd.Series], windows: dict[str, int] = RELATIVE_WINDOWS
) -> pd.DataFrame:
    """
    Trailing 1/3/6/12-month returns for the stock and each comparison, plus the stock's excess over each
    (stock return - comparison return), measured on common dates.
    """
    joined = pd.concat({"stock": stock, **others}, axis=1, join="inner").dropna()
    table = pd.DataFrame({name: trailing_returns(joined[name], windows) for name in joined.columns})
    for name in others:
        table[f"stock vs {name}"] = table["stock"] - table[name]
    return table


def rolling_relative(
    stock: pd.Series, benchmark: pd.Series, windows: dict[str, int] = RELATIVE_WINDOWS
) -> pd.DataFrame:
    """Rolling relative performance per window: stock's n-day return minus the benchmark's, on common dates."""
    joined = pd.concat({"stock": stock, "bench": benchmark}, axis=1, join="inner").dropna()
    return pd.DataFrame(
        {name: joined["stock"].pct_change(n) - joined["bench"].pct_change(n) for name, n in windows.items()}
    )
