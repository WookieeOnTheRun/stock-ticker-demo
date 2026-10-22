"""
Shared UI state. Every shared setting has one default here, is initialized once per session, and is kept when a page
that doesn't render its widget is shown (Streamlit otherwise drops values of widgets that weren't rendered).
"""

import streamlit as st

from stockapp.backtest.strategies import KELLY_METHODS
from stockapp.fx import BASE_CURRENCIES
from stockapp.userstore import BENCHMARK_CHOICES

HISTORY_OPTIONS: dict[str, int | None] = {"5y": 5, "10y": 10, "20y": 20, "max": None}
METHODS = list(KELLY_METHODS)
BENCHMARKS = list(BENCHMARK_CHOICES)
SURVIVORSHIP_NOTE = (
    "Analyses use currently listed tickers; delisted companies are absent, which biases historical comparisons upward."
)

DEFAULTS: dict[str, object] = {
    "ticker": "MSFT",
    "history": "10y",
    "base_currency": "USD",
    "benchmark_choice": "SPY",
    "benchmark_other": "SPY",
    "k_fraction": 0.5,
    "method": "Shrunk",
    "prior_sd_annual": 0.10,
    "max_dd_pct": 25,
    "lookback": 252,
}
"""Shared settings and their defaults (also the keys a signed-in user can save)."""


def init_state() -> None:
    """Set every shared default that isn't in the session yet, and keep widget values across page switches."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
        else:
            st.session_state[key] = st.session_state[key]  # interrupts Streamlit's cleanup of unrendered widgets


def years() -> int | None:
    """History length in years for the selected History option (None = all available)."""
    return HISTORY_OPTIONS[str(st.session_state["history"])]


def benchmark_text() -> str:
    """The benchmark ticker as typed or chosen (not yet validated)."""
    choice = str(st.session_state["benchmark_choice"])
    return str(st.session_state["benchmark_other"]) if choice == "Other" else choice


def render_sidebar() -> None:
    """Shared inputs, shown on every page. Widgets take their values from session state (no value= arguments)."""
    with st.sidebar:
        st.text_input("Ticker", key="ticker", placeholder="e.g., AAPL, MSFT, SPY, VOD.L", help=SURVIVORSHIP_NOTE)
        st.selectbox("History", list(HISTORY_OPTIONS), key="history")
        st.selectbox("Base currency", BASE_CURRENCIES, key="base_currency")
        st.selectbox("Benchmark", BENCHMARKS, key="benchmark_choice")
        if st.session_state["benchmark_choice"] == "Other":
            st.text_input("Benchmark ticker", key="benchmark_other", placeholder="e.g., VT, EFA, ^GSPC")
        st.slider("Kelly fraction multiplier", min_value=0.1, max_value=1.0, step=0.1, key="k_fraction")
        st.selectbox("Kelly estimate method", METHODS, key="method")
        st.number_input(
            "Prior SD of annual expected excess return",
            min_value=0.01,
            max_value=0.50,
            step=0.01,
            format="%.2f",
            key="prior_sd_annual",
        )
        st.caption(
            "Modelling assumption: before seeing data, the stock's annual expected excess return is believed to be "
            "within about ±2 prior SDs of zero. Smaller values shrink Kelly harder toward 0."
        )
        st.slider("Max tolerable drawdown", min_value=10, max_value=50, step=5, format="%d%%", key="max_dd_pct")
        st.caption("Probability of exceeding it over 1 year ≈ 10%.")


def current_settings() -> dict[str, object]:
    """The shared settings as they are now (the payload a signed-in user can save)."""
    return {key: st.session_state[key] for key in DEFAULTS}


def apply_settings(settings: dict[str, object]) -> None:
    """Overwrite shared settings; call from a widget callback so it runs before the sidebar widgets render."""
    for key, value in settings.items():
        if key in DEFAULTS:
            st.session_state[key] = value


def reset_to_defaults() -> None:
    """Restore every shared setting to DEFAULTS; call from a widget callback, like apply_settings."""
    apply_settings(dict(DEFAULTS))
