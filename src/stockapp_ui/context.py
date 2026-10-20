"""
The shared single-ticker pipeline: load prices in the base currency, the risk-free rate, returns over the lookback
window and the Tier 0 Kelly estimates. Pages that analyse one ticker call build_context(); it shows messages and
stops the page (st.stop) on unusable input, so the rest of each page can assume valid data.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from stockapp.bootstrap import KellyBand
from stockapp.calendars import is_stale, sessions_per_year
from stockapp.data import compute_returns, normalize_ticker
from stockapp.fx import convert_prices, fx_ticker, major_currency
from stockapp.kelly import KellyResult, excess_returns, fractional_kelly, shrunk_kelly
from stockapp.riskfree import align_rf, daily_rf
from stockapp_ui import cache, state


@dataclass
class ResearchContext:
    ticker: str
    history_label: str
    years: int | None
    base_currency: str
    method: str
    k_fraction: float
    prior_sd_annual: float
    max_dd_pct: int
    manual_rf_pct: float | None
    native_currency: str | None
    price_currency: str
    close_all: pd.Series
    close: pd.Series
    periods_per_year: float
    returns_all: pd.DataFrame
    returns_df: pd.DataFrame
    rf_all: pd.Series
    rf_df: pd.Series
    excess: pd.Series
    lookback: int
    kr: KellyResult
    raws: dict[str, float]
    gauss_band: KellyBand
    shrunk_band: KellyBand
    last_date: str
    benchmark: str | None
    benchmark_close: pd.Series | None
    benchmark_problem: str

    @property
    def max_dd(self) -> float:
        return self.max_dd_pct / 100

    @property
    def rf_key(self) -> str:
        return str(self.manual_rf_pct) if self.manual_rf_pct is not None else "irx"

    @property
    def data_key(self) -> str:
        """Cache key for analyses on the full loaded history (independent of the lookback slider)."""
        return f"{self.base_currency}|{self.history_label}|{self.rf_key}"

    @property
    def settings_key(self) -> str:
        """Cache key for analyses on the lookback window."""
        return f"{self.data_key}|{self.lookback}"

    @property
    def method_bands(self) -> dict[str, KellyBand]:
        return {"Gaussian": self.gauss_band, "Shrunk": self.shrunk_band}

    def raw_kelly(self, method_name: str, r: pd.Series, rf: pd.Series) -> float:
        """Unclamped Kelly f* for one method on returns r with matching risk-free rates rf."""
        if method_name == "Gaussian":
            return fractional_kelly(r, rf=rf).raw_kelly
        if method_name == "Shrunk":
            return shrunk_kelly(excess_returns(r, rf), self.prior_sd_annual)
        if method_name == "Log-optimal":
            return cache.log_optimal_cached(r, rf)
        return cache.drawdown_kelly_cached(r, rf, self.max_dd)

    def applied(self, raw: float) -> float:
        """Applied fraction: clip(fraction slider * raw f*, 0, 1). Long-only, no leverage."""
        return max(0.0, min(self.k_fraction * raw, 1.0))


def _manual_rf() -> float | None:
    st.warning("Couldn't fetch the T-bill rate. Enter a risk-free rate below; a manual rate is in use.")
    value = st.number_input(
        "Annual risk-free rate (%)", min_value=0.0, max_value=20.0, value=None, step=0.1, format="%.2f", key="manual_rf"
    )
    return None if value is None else float(value)


def build_context() -> ResearchContext:
    """Run the single-ticker pipeline for the shared sidebar settings; stops the page with a message if unusable."""
    ss = st.session_state
    history_label, base_currency = str(ss["history"]), str(ss["base_currency"])
    years = state.years()
    try:
        rf_annual_pct: pd.Series | None = cache.fetch_rf_cached(years)
    except Exception:
        rf_annual_pct = None
    manual_rf_pct: float | None = None
    if rf_annual_pct is None:
        manual_rf_pct = _manual_rf()
        if manual_rf_pct is None:
            st.stop()

    raw_ticker = str(ss["ticker"])
    if not raw_ticker.strip():
        st.info("Enter a ticker in the sidebar.")
        st.stop()
    ticker = normalize_ticker(raw_ticker)
    if ticker is None:
        st.error("Invalid ticker. Use letters, digits, and . ^ = - only (max 15 characters).")
        st.stop()

    try:
        native_close, native_currency = cache.load_close(ticker, years)
    except Exception as e:
        st.error(f"Could not load data for {ticker}: {e}")
        st.stop()

    # Convert to the base currency (minor units such as GBp -> GBP first). Returns, and so Kelly, are measured in it.
    major, divisor = major_currency(native_currency or base_currency)
    pair = None if major == base_currency else fx_ticker(major, base_currency)
    try:
        fx_rate = cache.load_fx(pair, years) if pair else None
        close_all = convert_prices(native_close, divisor, fx_rate)
        price_currency = base_currency
    except Exception:
        st.warning(f"Couldn't load the {pair} FX rate; prices are shown in {major}, not converted to {base_currency}.")
        close_all = convert_prices(native_close, divisor, None)
        price_currency = major
    if close_all.empty:
        st.error(f"No {pair} FX rates overlap {ticker}'s price history.")
        st.stop()
    if native_currency is None:
        st.caption(f"Quote currency unknown; prices assumed to be in {base_currency}.")
    elif native_currency == price_currency:
        st.caption(f"{ticker}: prices in {price_currency}.")
    else:
        st.caption(f"{ticker}: prices in {native_currency} (native), converted to {price_currency}.")

    if is_stale(ticker, close_all.index[-1].date(), date.today()):
        st.warning(f"Data for {ticker} may be stale: the last bar is from {close_all.index[-1].date()}.")
    periods_per_year = sessions_per_year(ticker, close_all.index[-1].date())

    returns_all = compute_returns(close_all)
    if rf_annual_pct is not None:
        rf_all = align_rf(daily_rf(rf_annual_pct), pd.DatetimeIndex(returns_all.index))
    else:
        rf_all = daily_rf(pd.Series(manual_rf_pct, index=returns_all.index))
    returns_all = returns_all.loc[rf_all.index]
    if len(returns_all) < 2:
        st.error(f"Not enough price history for {ticker} to compute returns.")
        st.stop()

    # The slider's range follows the loaded history; a floor of 70 keeps it usable for very short histories.
    max_lookback = max(len(returns_all), 70)
    ss["lookback"] = min(max(int(ss["lookback"]), 60), max_lookback)
    with st.sidebar:
        lookback = int(
            st.slider("Lookback window (trading days)", min_value=60, max_value=max_lookback, step=10, key="lookback")
        )
        st.caption("Data source: yfinance (adjusted close). No API key required.")
        st.caption("Risk-free: 13-week US T-bill (USD). Non-USD tickers should use a local rate (not yet supported).")

    close = close_all.tail(lookback + 1)
    returns_df = returns_all.tail(lookback)
    if len(returns_df) < lookback:
        st.warning(f"Only {len(returns_df)} trading days of history available for {ticker}.")

    rf_df = rf_all.loc[returns_df.index]
    excess = excess_returns(returns_df["simple_return"], rf_df)
    k_fraction, prior_sd_annual = float(ss["k_fraction"]), float(ss["prior_sd_annual"])
    max_dd = int(ss["max_dd_pct"]) / 100
    r = returns_df["simple_return"]
    raws = {
        "Gaussian": fractional_kelly(r, rf=rf_df).raw_kelly,
        "Shrunk": shrunk_kelly(excess, prior_sd_annual),
        "Log-optimal": cache.log_optimal_cached(r, rf_df),
        "Drawdown-constrained": cache.drawdown_kelly_cached(r, rf_df, max_dd),
    }
    gauss_band, shrunk_band = cache.bootstrap_bands(excess, prior_sd_annual)

    benchmark = normalize_ticker(state.benchmark_text())
    benchmark_close: pd.Series | None = None
    benchmark_problem = ""
    if benchmark is None:
        benchmark_problem = "Invalid benchmark ticker. Use letters, digits, and . ^ = - only (max 15 characters)."
    elif benchmark == ticker:
        benchmark_problem = f"The benchmark is {ticker} itself; choose a different benchmark."
    else:
        try:
            benchmark_close = cache.load_in_base(benchmark, years, base_currency)
        except Exception as e:
            benchmark_problem = f"Could not load benchmark {benchmark}: {e}"

    return ResearchContext(
        ticker=ticker,
        history_label=history_label,
        years=years,
        base_currency=base_currency,
        method=str(ss["method"]),
        k_fraction=k_fraction,
        prior_sd_annual=prior_sd_annual,
        max_dd_pct=int(ss["max_dd_pct"]),
        manual_rf_pct=manual_rf_pct,
        native_currency=native_currency,
        price_currency=price_currency,
        close_all=close_all,
        close=close,
        periods_per_year=periods_per_year,
        returns_all=returns_all,
        returns_df=returns_df,
        rf_all=rf_all,
        rf_df=rf_df,
        excess=excess,
        lookback=lookback,
        kr=fractional_kelly(r, k_fraction=k_fraction, rf=rf_df),
        raws=raws,
        gauss_band=gauss_band,
        shrunk_band=shrunk_band,
        last_date=str(close_all.index[-1].date()),
        benchmark=benchmark,
        benchmark_close=benchmark_close,
        benchmark_problem=benchmark_problem,
    )
