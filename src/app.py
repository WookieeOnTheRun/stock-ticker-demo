import pandas as pd
import streamlit as st

from stockapp.bootstrap import KellyBand, block_bootstrap_indices, kelly_band, kelly_distribution
from stockapp.charts import cumulative_return_chart, kelly_over_windows_chart, price_chart, returns_histogram
from stockapp.data import compute_returns, fetch_price_history, normalize_ticker
from stockapp.drawdown import drawdown_constrained_kelly
from stockapp.kelly import excess_returns, fractional_kelly, log_optimal_kelly, shrunk_kelly
from stockapp.metrics import annualized_volatility, cagr, growth_rate, sharpe_ratio
from stockapp.riskfree import align_rf, daily_rf, fetch_rf_annual_pct

WINDOW_SIZES = [60, 120, 252, 504]

# Fetch once per ticker per hour; slider changes only slice the cached data.
fetch_cached = st.cache_data(ttl=3600, show_spinner="Fetching price history...")(fetch_price_history)
fetch_rf_cached = st.cache_data(ttl=3600, show_spinner="Fetching risk-free rate...")(fetch_rf_annual_pct)
log_optimal_cached = st.cache_data(show_spinner=False)(log_optimal_kelly)
drawdown_kelly_cached = st.cache_data(show_spinner=False)(drawdown_constrained_kelly)


@st.cache_data(show_spinner=False)
def bootstrap_bands(excess: pd.Series, prior_sd_annual: float) -> tuple[KellyBand, KellyBand]:
    """Bootstrap 5-95% bands for (Gaussian, Shrunk) Kelly on the same resamples."""
    idx = block_bootstrap_indices(len(excess))
    return (
        kelly_band(kelly_distribution(excess, idx)),
        kelly_band(kelly_distribution(excess, idx, prior_sd_annual)),
    )


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
    max_dd_pct = st.slider("Max tolerable drawdown", min_value=10, max_value=50, value=25, step=5, format="%d%%")
    st.caption("Probability of exceeding it over 1 year ≈ 10%.")
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
gauss_band, shrunk_band = bootstrap_bands(excess, prior_sd_annual)
dd_raw = drawdown_kelly_cached(returns_df["simple_return"], rf_df, max_dd_pct / 100)

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
metrics_col1.metric("CAGR", f"{cagr(close):.1%}")
metrics_col2.metric("Annualized volatility", f"{annualized_volatility(returns_df['simple_return']):.1%}")
metrics_col3.metric("Sharpe ratio (excess)", f"{sharpe_ratio(excess):.2f}")
metrics_col4.metric("Growth at applied fraction", f"{growth_rate(kr.applied_kelly, rf_df, excess):.1%}")
st.caption("Growth at applied fraction is a second-order approximation: g(f) ≈ 252·[mean(rf) + f·mean(e) − f²σ²/2].")

st.metric(label="Applied Fractional Kelly", value=f"{kr.applied_kelly:.4f}", delta=f"fraction={kr.fraction:.2f}")

with st.expander("Advanced"):
    adv1, adv2, adv3, adv4 = st.columns(4)
    adv1.metric("Mean excess return (daily)", f"{kr.mean_return:.6f}")
    adv2.metric("Variance (daily)", f"{kr.variance:.6f}")
    adv3.metric("Std dev (daily)", f"{kr.stdev:.6f}")
    adv4.metric("Mean risk-free (daily)", f"{rf_df.mean():.6f}")
    adv5, adv6, adv7, adv8 = st.columns(4)
    adv5.metric("Raw Kelly ((mu-r)/var)", f"{kr.raw_kelly:.4f}")
    adv6.metric("Shrunk Kelly (raw)", f"{shrunk_raw:.4f}")
    adv7.metric("Log-optimal Kelly (raw)", f"{log_opt_raw:.4f}")
    adv8.metric(f"Drawdown-constrained (max DD {max_dd_pct}%)", f"{dd_raw:.2f}")
    for name, band in (("Gaussian", gauss_band), ("Shrunk", shrunk_band)):
        st.caption(
            f"{name} Kelly bootstrap: 5th {band.p5:.2f} · median {band.p50:.2f} · 95th {band.p95:.2f} · "
            f"P(f* > 0) = {band.prob_positive:.0%}"
        )

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
