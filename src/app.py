import streamlit as st
from stockapp.data import fetch_price_history, compute_returns, normalize_ticker
from stockapp.kelly import fractional_kelly
from stockapp.charts import price_chart, returns_histogram, cumulative_return_chart, kelly_over_windows_chart

WINDOW_SIZES = [60, 120, 252, 504]

# Fetch once per ticker per hour; slider changes only slice the cached data.
fetch_cached = st.cache_data(ttl=3600, show_spinner="Fetching price history...")(fetch_price_history)

st.set_page_config(page_title="Fractional Kelly Analyzer", layout="wide")

st.title("📈 Fractional Kelly Stock Analyzer")

col_left, col_right = st.columns([3, 2])

with col_left:
    raw_ticker = st.text_input("Ticker", value="MSFT", placeholder="e.g., AAPL, MSFT, SPY")
    lookback = st.slider("Lookback window (trading days)", min_value=60, max_value=1000, value=252, step=10)
    k_fraction = st.slider("Kelly fraction multiplier", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    st.caption("Data source: yfinance (adjusted close). No API key required.")

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
if len(returns_all) < 2:
    st.error(f"Not enough price history for {ticker} to compute returns.")
    st.stop()

close = close_all.tail(lookback + 1)
returns_df = returns_all.tail(lookback)
if len(returns_df) < lookback:
    st.warning(f"Only {len(returns_df)} trading days of history available for {ticker}.")

kr = fractional_kelly(returns_df["simple_return"], k_fraction=k_fraction)

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
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
metrics_col1.metric("Mean Return (daily)", f"{kr.mean_return:.6f}")
metrics_col2.metric("Variance", f"{kr.variance:.6f}")
metrics_col3.metric("Std Dev", f"{kr.stdev:.6f}")
metrics_col4.metric("Raw Kelly (mu/var)", f"{kr.raw_kelly:.4f}")

st.metric(label="Applied Fractional Kelly", value=f"{kr.applied_kelly:.4f}", delta=f"fraction={kr.fraction:.2f}")

# Kelly across fixed windows of full history (independent of the lookback slider); skip windows longer than the data
kelly_by_window = {
    w: fractional_kelly(returns_all["simple_return"].tail(w), k_fraction=k_fraction).applied_kelly
    for w in WINDOW_SIZES
    if w <= len(returns_all)
}
st.plotly_chart(kelly_over_windows_chart(kelly_by_window))

# Data preview and download
st.subheader("Data")
st.dataframe(returns_df.tail(20))
csv = returns_df.to_csv().encode("utf-8")
st.download_button("Download returns CSV", data=csv, file_name=f"{ticker}_returns.csv", mime="text/csv")
