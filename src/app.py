import streamlit as st
from src.data import fetch_price_history, compute_returns
from src.kelly import fractional_kelly
from src.charts import price_chart, returns_histogram, cumulative_return_chart, kelly_over_windows_chart

st.set_page_config(page_title="Fractional Kelly Analyzer", layout="wide")

st.title("📈 Fractional Kelly Stock Analyzer")

col_left, col_right = st.columns([3, 2])

with col_left:
    ticker = st.text_input("Ticker", value="AAPL", placeholder="e.g., AAPL, MSFT, SPY")
    lookback = st.slider("Lookback window (days)", min_value=60, max_value=1000, value=252, step=10)
    k_fraction = st.slider("Kelly fraction multiplier", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    st.caption("Data source: yfinance (close). No API key required.")

if ticker:
    price_df = fetch_price_history(ticker, lookback)
    if price_df.empty:
        st.error("No data returned. Check ticker or try a larger lookback.")
        st.stop()

    returns_df = compute_returns(price_df)
    kr = fractional_kelly(returns_df["simple_return"], k_fraction=k_fraction)

    # Charts
    st.subheader("Charts")
    p1, p2 = st.tabs(["Price", "Returns"])
    with p1:
        st.plotly_chart(price_chart(price_df), use_container_width=True)
    with p2:
        st.plotly_chart(returns_histogram(returns_df), use_container_width=True)
        st.plotly_chart(cumulative_return_chart(returns_df), use_container_width=True)

    # Kelly breakdown
    st.subheader("Kelly Breakdown")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    metrics_col1.metric("Mean Return (daily)", f"{kr.mean_return:.6f}")
    metrics_col2.metric("Variance", f"{kr.variance:.6f}")
    metrics_col3.metric("Std Dev", f"{kr.stdev:.6f}")
    metrics_col4.metric("Raw Kelly (mu/var)", f"{kr.raw_kelly:.4f}")

    st.metric(label="Applied Fractional Kelly", value=f"{kr.applied_kelly:.4f}", delta=f"fraction={kr.fraction:.2f}")

    # Kelly across windows
    window_sizes = [60, 120, 252, 504]
    st.plotly_chart(kelly_over_windows_chart(returns_df, window_sizes, lambda s: fractional_kelly(s, k_fraction=k_fraction)), use_container_width=True)

    # Data preview and download
    st.subheader("Data")
    st.dataframe(returns_df.tail(20))
    csv = returns_df.to_csv().encode("utf-8")
    st.download_button("Download returns CSV", data=csv, file_name=f"{ticker}_returns.csv", mime="text/csv")
