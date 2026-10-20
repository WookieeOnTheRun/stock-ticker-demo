"""Streamlit UI for the Portfolio tab. Analytics live in stockapp.*; this module only gathers input and renders."""

from collections.abc import Callable

import pandas as pd
import streamlit as st

from stockapp.portfolio_data import InsufficientDataError, ReturnsData, default_frequency, returns_matrix
from stockapp.portfolio_input import (
    MAX_TICKERS,
    MAX_UPLOAD_BYTES,
    Watchlist,
    WatchlistError,
    parse_ticker_text,
    parse_watchlist_csv,
    shares_to_weights,
)

DEFAULT_TICKERS = "AAPL, MSFT, JNJ, XOM, JPM, SPY"

LoadInBase = Callable[[str, int | None, str], pd.Series]
DailyRf = Callable[[], pd.Series | None]


def _watchlist_input() -> Watchlist | None:
    source = st.radio("Watchlist", ["Type tickers", "Upload CSV"], horizontal=True, key="pf_source")
    if source == "Type tickers":
        text = st.text_area(
            f"Tickers (up to {MAX_TICKERS}, separated by commas or spaces)", DEFAULT_TICKERS, key="pf_tickers"
        )
        try:
            return Watchlist(parse_ticker_text(text))
        except WatchlistError as exc:
            st.error(str(exc))
            return None
    upload = st.file_uploader(
        "CSV with a 'ticker' column and optionally 'weight' or 'shares' (max 1 MB, 50 tickers)",
        type=["csv"],
        max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
        key="pf_upload",
    )
    if upload is None:
        st.caption("Example: ticker,weight / MSFT,0.30 / AAPL,0.20 — weights are fractions; the rest is cash.")
        return None
    try:
        return parse_watchlist_csv(upload.getvalue())
    except WatchlistError as exc:
        st.error(str(exc))
        return None


def _risk_free(daily_rf: DailyRf) -> pd.Series | None:
    """Daily T-bill rate, or a constant manual rate entered here if the T-bill fetch failed."""
    rf = daily_rf()
    if rf is not None:
        return rf
    st.warning("Couldn't fetch the T-bill rate. Enter a risk-free rate; a manual rate is in use.")
    manual = st.number_input(
        "Annual risk-free rate (%)", min_value=0.0, max_value=20.0, value=None, step=0.1, format="%.2f", key="pf_rf"
    )
    if manual is None:
        return None
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * 60)
    return pd.Series((1 + manual / 100) ** (1 / 252) - 1, index=index)


def render(load_in_base: LoadInBase, daily_rf: DailyRf, years: int | None) -> None:
    """Render the Portfolio tab. Settings shared with the Stock tab are read from st.session_state."""
    base_currency = str(st.session_state.get("base_currency", "USD"))
    st.caption(f"Uses the Stock tab's base currency ({base_currency}), Kelly fraction and prior SD settings.")
    watchlist = _watchlist_input()
    if watchlist is None or not watchlist.tickers:
        return
    if len(watchlist.tickers) < 2:
        st.info("Enter at least two tickers to build a portfolio.")
        return

    prices: dict[str, pd.Series] = {}
    failed: list[str] = []
    for ticker in watchlist.tickers:
        try:
            prices[ticker] = load_in_base(ticker, years, base_currency)
        except Exception:
            failed.append(ticker)
    if failed:
        st.warning(f"Could not load {', '.join(failed)}; they are left out.")
    if len(prices) < 2:
        st.info("At least two tickers with price data are needed.")
        return

    latest = {t: float(p.iloc[-1]) for t, p in prices.items()}
    current: dict[str, float] = {t: 0.0 for t in prices}
    holdings_value: float | None = None
    if watchlist.weights is not None:
        current.update({t: w for t, w in watchlist.weights.items() if t in prices})
    elif watchlist.shares is not None:
        weights, holdings_value = shares_to_weights(watchlist.shares, latest)
        current.update(weights)
    st.dataframe(
        pd.DataFrame(
            {
                f"Latest price ({base_currency})": latest,
                "Current weight": current,
                "History from": {t: p.index[0].date() for t, p in prices.items()},
            }
        )
    )
    if holdings_value is not None:
        st.caption(f"Holdings value from shares × latest prices: {holdings_value:,.2f} {base_currency}.")

    rf = _risk_free(daily_rf)
    if rf is None:
        return
    frequencies = ["Daily", "Weekly"]
    suggested = default_frequency(list(prices))
    frequency = st.selectbox(
        "Return frequency", frequencies, index=frequencies.index(suggested), key=f"pf_freq_{suggested}"
    )
    st.caption(
        "Exchanges close at different times (e.g. New York vs Hong Kong), so their daily returns cover different "
        "windows and daily correlations are biased toward zero. Weekly returns largely remove this; the default is "
        "Weekly when the tickers span different exchanges. Weekly data is annualized with 52 periods."
    )
    try:
        data: ReturnsData = returns_matrix(prices, rf, "Weekly" if frequency == "Weekly" else "Daily")
    except InsufficientDataError as exc:
        st.error(str(exc))
        return
    st.caption(
        f"{data.common_days} trading dates are common to all {len(prices)} tickers "
        f"({len(data.returns)} {frequency.lower()} returns)."
    )
