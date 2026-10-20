import pandas as pd
import streamlit as st

from stockapp.charts import cumulative_return_chart, kelly_over_windows_chart, price_chart, returns_histogram
from stockapp.data import compute_returns, fetch_price_history, normalize_ticker
from stockapp.kelly import excess_returns, fractional_kelly, log_optimal_kelly, shrunk_kelly
from stockapp.riskfree import align_rf, daily_rf, fetch_rf_annual_pct

WINDOW_SIZES = [60, 120, 252, 504]

# Fetch once per ticker per hour; slider changes only slice the cached data.
fetch_cached = st.cache_data(ttl=3600, show_spinner="Fetching price history...")(fetch_price_history)
fetch_rf_cached = st.cache_data(ttl=3600, show_spinner="Fetching risk-free rate...")(fetch_rf_annual_pct)
log_optimal_cached = st.cache_data(show_spinner=False)(log_optimal_kelly)

st.set_page_config(page_title="Fractional Kelly Analyzer", layout="wide")

st.title("📈 Fractional Kelly Stock Analyzer")

col_left, col_right = st.columns([3, 2])

with col_left:
    raw_ticker = st.text_input("Ticker", value="MSFT", placeholder="e.g., AAPL, MSFT, SPY")
    lookback = st.slider("Lookback window (trading days)", min_value=60, max_value=1000, value=252, step=10)
    k_fraction = st.slider("Kelly fraction multiplier", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    prior_sd_annual = st.number_input(
        "Prior SD of annual expected excess return",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        format="%.2f",
    )
    st.caption(
        "Modelling assumption: before seeing data, the stock's annual expected excess return is believed to be "
        "within about ±2 prior SDs of zero. Smaller values shrink Kelly harder toward 0."
    )
    st.caption("Data source: yfinance (adjusted close). No API key required.")
    st.caption("Risk-free: 13-week US T-bill (USD). Non-USD tickers should use a local rate (not yet supported).")

try:
    rf_annual_pct: pd.Series | None = fetch_rf_cached()
except Exception:
    rf_annual_pct = None

manual_rf_pct: float | None = None
if rf_annual_pct is None:
    with col_left:
        st.warning("Couldn't fetch the T-bill rate. Enter a risk-free rate below; a manual rate is in use.")
        manual_rf_pct = st.number_input(
            "Annual risk-free rate (%)", min_value=0.0, max_value=20.0, value=None, step=0.1, format="%.2f"
        )
    if manual_rf_pct is None:
        st.stop()

if not raw_ticker.strip():
    st.stop()

ticker = normalize_ticker(raw_ticker)
if ticker is None:
    st.error("Invalid ticker. Use letters, digits, and . ^ = - only (max 15 characters).")
    st.stop()

try:
    close_all = fetch_cached(ticker)
except Exception as e:
    st.error(f"Could not load data for {ticker}: {e}")
    st.stop()

returns_all = compute_returns(close_all)
if rf_annual_pct is not None:
    rf_all = align_rf(daily_rf(rf_annual_pct), returns_all.index)
else:
    rf_all = daily_rf(pd.Series(manual_rf_pct, index=returns_all.index))
returns_all = returns_all.loc[rf_all.index]
if len(returns_all) < 2:
    st.error(f"Not enough price history for {ticker} to compute returns.")
    st.stop()

close = close_all.tail(lookback + 1)
returns_df = returns_all.tail(lookback)
if len(returns_df) < lookback:
    st.warning(f"Only {len(returns_df)} trading days of history available for {ticker}.")

rf_df = rf_all.loc[returns_df.index]
excess = excess_returns(returns_df["simple_return"], rf_df)
kr = fractional_kelly(returns_df["simple_return"], k_fraction=k_fraction, rf=rf_df)
shrunk_raw = shrunk_kelly(excess, prior_sd_annual)
log_opt_raw = log_optimal_cached(returns_df["simple_return"], rf_df)

# Charts
st.subheader("Charts")
p1, p2 = st.tabs(["Price", "Returns"])
with p1:
    st.plotly_chart(price_chart(close))
with p2:
    st.plotly_chart(returns_histogram(returns_df))
    st.plotly_chart(cumulative_return_chart(returns_df))

# Kelly breakdown
st.subheader("Kelly Breakdown")
metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5, metrics_col6 = st.columns(6)
metrics_col1.metric("Mean Excess Return (daily)", f"{kr.mean_return:.6f}")
metrics_col2.metric("Variance", f"{kr.variance:.6f}")
metrics_col3.metric("Std Dev", f"{kr.stdev:.6f}")
metrics_col4.metric("Raw Kelly ((mu-r)/var)", f"{kr.raw_kelly:.4f}")
metrics_col5.metric("Shrunk Kelly (raw)", f"{shrunk_raw:.4f}")
metrics_col6.metric("Log-optimal Kelly (raw)", f"{log_opt_raw:.4f}")

st.metric(label="Applied Fractional Kelly", value=f"{kr.applied_kelly:.4f}", delta=f"fraction={kr.fraction:.2f}")

# Kelly across fixed windows of full history (independent of the lookback slider); skip windows longer than the data
kelly_by_window = {
    w: fractional_kelly(returns_all["simple_return"].tail(w), k_fraction=k_fraction, rf=rf_all).applied_kelly
    for w in WINDOW_SIZES
    if w <= len(returns_all)
}
st.plotly_chart(kelly_over_windows_chart(kelly_by_window))

# Data preview and download
st.subheader("Data")
st.dataframe(returns_df.tail(20))
csv = returns_df.to_csv().encode("utf-8")
st.download_button("Download returns CSV", data=csv, file_name=f"{ticker}_returns.csv", mime="text/csv")
