from datetime import date

import pandas as pd
import streamlit as st

from stockapp.bootstrap import KellyBand, block_bootstrap_indices, kelly_band, kelly_distribution
from stockapp.calendars import is_stale, sessions_per_year
from stockapp.charts import cumulative_return_chart, kelly_over_windows_chart, price_chart, returns_histogram
from stockapp.data import compute_returns, normalize_ticker
from stockapp.drawdown import drawdown_constrained_kelly
from stockapp.kelly import excess_returns, fractional_kelly, log_optimal_kelly, shrunk_kelly
from stockapp.metrics import annualized_volatility, cagr, growth_rate, sharpe_ratio
from stockapp.repository import load_bars
from stockapp.riskfree import align_rf, daily_rf, fetch_rf_annual_pct

WINDOW_SIZES = [60, 120, 252, 504]
METHODS = ["Gaussian", "Shrunk", "Log-optimal", "Drawdown-constrained"]
HISTORY_OPTIONS: dict[str, int | None] = {"5y": 5, "10y": 10, "20y": 20, "max": None}
BOOTSTRAP_METHODS = {"Gaussian", "Shrunk"}

# Short in-memory layer over the SQLite cache (stockapp.repository); slider changes only slice the cached data.
fetch_rf_cached = st.cache_data(ttl=300, show_spinner="Loading risk-free rate...")(fetch_rf_annual_pct)
log_optimal_cached = st.cache_data(show_spinner=False)(log_optimal_kelly)
drawdown_kelly_cached = st.cache_data(show_spinner=False)(drawdown_constrained_kelly)


@st.cache_data(ttl=300, show_spinner="Loading price history...")
def load_close(ticker: str, years: int | None) -> pd.Series:
    """Adjusted closes named "Close" for the last `years` years (None = all available)."""
    return load_bars(ticker, years)["adj_close"].rename("Close")


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
    history_label = st.selectbox("History", list(HISTORY_OPTIONS), index=list(HISTORY_OPTIONS).index("10y"))
    lookback_slot = st.empty()  # filled once the available history is known
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
    method = st.selectbox("Kelly estimate method", METHODS, index=METHODS.index("Shrunk"))
    max_dd_pct = st.slider("Max tolerable drawdown", min_value=10, max_value=50, value=25, step=5, format="%d%%")
    st.caption("Probability of exceeding it over 1 year ≈ 10%.")
    st.caption("Data source: yfinance (adjusted close). No API key required.")
    st.caption("Risk-free: 13-week US T-bill (USD). Non-USD tickers should use a local rate (not yet supported).")

try:
    rf_annual_pct: pd.Series | None = fetch_rf_cached(HISTORY_OPTIONS[history_label])
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
    close_all = load_close(ticker, HISTORY_OPTIONS[history_label])
except Exception as e:
    st.error(f"Could not load data for {ticker}: {e}")
    st.stop()

if is_stale(ticker, close_all.index[-1].date(), date.today()):
    st.warning(f"Data for {ticker} may be stale: the last bar is from {close_all.index[-1].date()}.")
periods_per_year = sessions_per_year(ticker, close_all.index[-1].date())

returns_all = compute_returns(close_all)
if rf_annual_pct is not None:
    rf_all = align_rf(daily_rf(rf_annual_pct), returns_all.index)
else:
    rf_all = daily_rf(pd.Series(manual_rf_pct, index=returns_all.index))
returns_all = returns_all.loc[rf_all.index]
if len(returns_all) < 2:
    st.error(f"Not enough price history for {ticker} to compute returns.")
    st.stop()

# The slider's range follows the loaded history; a floor of 70 keeps it usable for very short histories.
max_lookback = max(len(returns_all), 70)
lookback = lookback_slot.slider(
    "Lookback window (trading days)", min_value=60, max_value=max_lookback, value=min(252, max_lookback), step=10
)

close = close_all.tail(lookback + 1)
returns_df = returns_all.tail(lookback)
if len(returns_df) < lookback:
    st.warning(f"Only {len(returns_df)} trading days of history available for {ticker}.")

rf_df = rf_all.loc[returns_df.index]
excess = excess_returns(returns_df["simple_return"], rf_df)
kr = fractional_kelly(returns_df["simple_return"], k_fraction=k_fraction, rf=rf_df)
max_dd = max_dd_pct / 100


def raw_kelly(method_name: str, r: pd.Series, rf: pd.Series) -> float:
    """Unclamped Kelly f* for one method on returns r with matching risk-free rates rf."""
    if method_name == "Gaussian":
        return fractional_kelly(r, rf=rf).raw_kelly
    if method_name == "Shrunk":
        return shrunk_kelly(excess_returns(r, rf), prior_sd_annual)
    if method_name == "Log-optimal":
        return log_optimal_cached(r, rf)
    return drawdown_kelly_cached(r, rf, max_dd)


def applied(raw: float) -> float:
    """Applied fraction: clip(fraction slider * raw f*, 0, 1). Long-only, no leverage."""
    return max(0.0, min(k_fraction * raw, 1.0))


raws = {m: raw_kelly(m, returns_df["simple_return"], rf_df) for m in METHODS}
gauss_band, shrunk_band = bootstrap_bands(excess, prior_sd_annual)
method_bands = {"Gaussian": gauss_band, "Shrunk": shrunk_band}
applied_f = applied(raws[method])

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
metrics_col1.metric("CAGR", f"{cagr(close, periods_per_year):.1%}")
metrics_col2.metric(
    "Annualized volatility", f"{annualized_volatility(returns_df['simple_return'], periods_per_year):.1%}"
)
metrics_col3.metric("Sharpe ratio (excess)", f"{sharpe_ratio(excess, periods_per_year):.2f}")
metrics_col4.metric(
    f"Growth at applied fraction ({method})", f"{growth_rate(applied_f, rf_df, excess, periods_per_year):.1%}"
)
st.caption(
    "Growth at applied fraction is a second-order approximation: g(f) ≈ P·[mean(rf) + f·mean(e) − f²σ²/2], "
    "with P trading sessions per year."
)

st.metric(label=f"Applied Fractional Kelly ({method})", value=f"{applied_f:.4f}", delta=f"fraction={kr.fraction:.2f}")

if gauss_band.crosses_zero:
    st.info(
        "The data can't distinguish this stock's expected excess return from zero; Kelly sizing is unreliable here."
    )

notes = {
    "Gaussian": f"(μ − r) / σ² of excess returns. P(f* > 0) = {gauss_band.prob_positive:.0%}.",
    "Shrunk": f"Mean shrunk toward 0 by its noise (prior SD {prior_sd_annual:.0%}). P(f* > 0) = "
    f"{shrunk_band.prob_positive:.0%}.",
    "Log-optimal": "Maximizes historical mean log growth; no Gaussian assumption (fat-tail aware).",
    "Drawdown-constrained": f"Largest f with a 90% chance of max drawdown ≤ {max_dd_pct}% over 1 year.",
}
comparison = pd.DataFrame(
    {
        "Method": METHODS,
        "Raw f*": [f"{raws[m]:.3f}" for m in METHODS],
        "Applied f": [f"{applied(raws[m]):.3f}" for m in METHODS],
        "Bootstrap 5–95% (raw f*)": [
            f"{method_bands[m].p5:.2f} to {method_bands[m].p95:.2f}" if m in method_bands else "—" for m in METHODS
        ],
        "Note": [notes[m] for m in METHODS],
    }
).set_index("Method")
st.table(comparison)

with st.expander("Advanced"):
    adv1, adv2, adv3, adv4 = st.columns(4)
    adv1.metric("Mean excess return (daily)", f"{kr.mean_return:.6f}")
    adv2.metric("Variance (daily)", f"{kr.variance:.6f}")
    adv3.metric("Std dev (daily)", f"{kr.stdev:.6f}")
    adv4.metric("Mean risk-free (daily)", f"{rf_df.mean():.6f}")
    st.caption(f"Annualization uses {periods_per_year:.0f} trading sessions per year for {ticker}'s exchange.")
    for name, band in method_bands.items():
        st.caption(
            f"{name} Kelly bootstrap: 5th {band.p5:.2f} · median {band.p50:.2f} · 95th {band.p95:.2f} · "
            f"P(f* > 0) = {band.prob_positive:.0%}"
        )

# Kelly across fixed windows of full history (independent of the lookback slider); skip windows longer than the data
kelly_by_window: dict[int, float] = {}
window_bands: dict[int, tuple[float, float]] = {}
for w in WINDOW_SIZES:
    if w > len(returns_all):
        continue
    r_w = returns_all["simple_return"].tail(w)
    rf_w = rf_all.loc[r_w.index]
    kelly_by_window[w] = applied(raw_kelly(method, r_w, rf_w))
    if method in BOOTSTRAP_METHODS:
        g_band, s_band = bootstrap_bands(excess_returns(r_w, rf_w), prior_sd_annual)
        band = g_band if method == "Gaussian" else s_band
        window_bands[w] = (applied(band.p5), applied(band.p95))
st.plotly_chart(
    kelly_over_windows_chart(
        kelly_by_window, bands=window_bands, title=f"Applied Kelly Across Lookback Windows ({method})"
    )
)
if method in BOOTSTRAP_METHODS:
    st.caption("Error bars: bootstrap 5–95% range of the applied fraction.")

# Data preview and download
st.subheader("Data")
export_df = returns_df.assign(rf_daily=rf_df, excess_return=excess)
st.dataframe(export_df.tail(20))
csv = export_df.to_csv().encode("utf-8")
st.download_button("Download returns CSV", data=csv, file_name=f"{ticker}_returns.csv", mime="text/csv")
