"""Streamlit cache layer over stockapp: short in-memory caches over the SQLite store and cached analytics."""

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stockapp import distribution, factors, regimes, risk
from stockapp.advice import fan_chart_paths, rule_strategy
from stockapp.backtest.engine import BacktestResult
from stockapp.backtest.strategies import Strategy, kelly_buy_and_hold
from stockapp.backtest.walkforward import WalkForwardResult, fixed_params, walk_forward
from stockapp.bootstrap import KellyBand, block_bootstrap_indices, kelly_band, kelly_distribution
from stockapp.drawdown import drawdown_constrained_kelly
from stockapp.fx import convert_prices, fx_ticker, major_currency
from stockapp.kelly import log_optimal_kelly
from stockapp.model.probability import ModelReport, walk_forward_model
from stockapp.providers.yfinance_provider import YFinanceProvider
from stockapp.repository import load_actions, load_bars, load_currency
from stockapp.riskfree import daily_rf, fetch_rf_annual_pct
from stockapp.signals import SIGNALS
from stockapp_ui import state

# Short in-memory layer over the SQLite cache (stockapp.repository); slider changes only slice the cached data.
fetch_rf_cached = st.cache_data(ttl=300, show_spinner="Loading risk-free rate...")(fetch_rf_annual_pct)
log_optimal_cached = st.cache_data(show_spinner=False)(log_optimal_kelly)
drawdown_kelly_cached = st.cache_data(show_spinner=False)(drawdown_constrained_kelly)


@st.cache_data(ttl=300, show_spinner="Loading price history...")
def load_close(ticker: str, years: int | None) -> tuple[pd.Series, str | None]:
    """Adjusted closes named "Close" for the last `years` years (None = all available), and the quote currency."""
    return load_bars(ticker, years)["adj_close"].rename("Close"), load_currency(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def load_actions_cached(ticker: str) -> pd.DataFrame:
    """Cached splits and dividends for a ticker already loaded with load_bars."""
    return load_actions(ticker)


@st.cache_data(ttl=300, show_spinner="Loading FX rates...")
def load_fx(pair: str, years: int | None) -> pd.Series:
    """FX closes for a yfinance pair such as GBPUSD=X."""
    return load_bars(pair, years)["close"]


@st.cache_data(ttl=300, show_spinner=False)
def load_in_base(ticker: str, years: int | None, base: str) -> pd.Series:
    """A comparison ticker's adjusted closes converted to the base currency (raises if prices or FX are missing)."""
    native, currency = load_close(ticker, years)
    major_ccy, minor_divisor = major_currency(currency or base)
    rate = load_fx(fx_ticker(major_ccy, base), years) if major_ccy != base else None
    return convert_prices(native, minor_divisor, rate)


@st.cache_data(ttl=86400, show_spinner=False)
def sector_cached(ticker: str) -> str | None:
    """yfinance sector for a ticker, cached for a day."""
    return YFinanceProvider().get_sector(ticker)


@st.cache_data(ttl=86400, show_spinner="Loading factor data...")
def factors_cached() -> pd.DataFrame:
    """Fama-French 5 + momentum factors (SQLite cache, refreshed weekly)."""
    return factors.load_factors()


# Analytics caches are keyed on (ticker, last bar date, settings key, params); data arguments start with "_" so
# Streamlit does not hash them.
@st.cache_data(show_spinner=False)
def risk_cached(
    ticker: str, last_date: str, key: str, ppy: float, _r: pd.Series, _close: pd.Series, _rf: pd.Series
) -> risk.RiskSummary:
    return risk.summarize(_r, _close, _rf, ppy)


@st.cache_data(show_spinner="Fitting GARCH(1,1)...")
def garch_cached(ticker: str, last_date: str, key: str, ppy: float, _r: pd.Series) -> risk.GarchForecast | None:
    try:
        return risk.garch_vol_forecast(_r, periods_per_year=ppy)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def distribution_cached(ticker: str, last_date: str, key: str, _r: pd.Series) -> distribution.DistributionSummary:
    return distribution.summarize(_r)


@st.cache_data(show_spinner=False)
def regimes_cached(
    ticker: str, last_date: str, key: str, prior: float, _r: pd.Series, _rf: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    labels = regimes.vol_regimes(_r)
    return labels, regimes.kelly_by_regime(_r, _rf, labels, prior)


@st.cache_data(show_spinner="Fitting HMM...")
def hmm_cached(ticker: str, last_date: str, key: str, n_states: int, _r: pd.Series) -> pd.Series | None:
    try:
        return regimes.hmm_regimes(_r, n_states)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def factor_fit_cached(
    ticker: str, last_date: str, key: str, factors_through: str, _r: pd.Series, _factors: pd.DataFrame
) -> factors.FactorFit:
    return factors.factor_regression(factors.stock_excess_returns(_r, _factors), _factors)


def strategy_by_name(name: str, kelly_settings: tuple[str, float, float, float]) -> Strategy:
    """A registered signal, or Kelly buy & hold configured with the sidebar's Kelly settings."""
    return kelly_buy_and_hold(*kelly_settings) if name == "Kelly buy & hold" else SIGNALS[name]


@st.cache_data(show_spinner="Running walk-forward backtest...")
def walk_forward_cached(
    ticker: str,
    last_date: str,
    key: str,
    name: str,
    kelly_settings: tuple[str, float, float, float],
    cost_bps: float,
    slippage_bps: float,
    mode: str,
    ppy: float,
    _prices: pd.Series,
    _rf: pd.Series,
) -> WalkForwardResult:
    return walk_forward(
        strategy_by_name(name, kelly_settings),
        _prices,
        _rf,
        cost_bps,
        slippage_bps,
        mode="rolling" if mode == "rolling" else "expanding",
        periods_per_year=ppy,
    )


@st.cache_data(show_spinner="Training walk-forward probability model...")
def model_cached(
    ticker: str,
    last_date: str,
    key: str,
    horizon: int,
    benchmark: str | None,
    _prices: pd.Series,
    _rf: pd.Series,
    _benchmark_prices: pd.Series | None,
) -> ModelReport | str:
    """The walk-forward model report, or a message explaining why it could not be built."""
    try:
        return walk_forward_model(_prices, _rf, horizon, _benchmark_prices)
    except ValueError as exc:
        return str(exc)


@st.cache_data(show_spinner="Backtesting the advice rule...")
def rule_track_record_cached(
    ticker: str,
    last_date: str,
    key: str,
    horizon: int,
    rule_settings: tuple[float, float, float, float, float],
    cost_bps: float,
    ppy: float,
    _p_used: pd.Series,
    _prices: pd.Series,
    _rf: pd.Series,
) -> WalkForwardResult:
    k_frac, prior, dd, p_buy, p_sell = rule_settings
    return walk_forward(
        rule_strategy(_p_used, k_frac, prior, dd, p_buy, p_sell), _prices, _rf, cost_bps, periods_per_year=ppy
    )


@st.cache_data(show_spinner=False)
def fan_cached(
    ticker: str, last_date: str, key: str, fraction: float, _prices: pd.Series, _rf: pd.Series
) -> pd.DataFrame:
    return fan_chart_paths(_prices, _rf, fraction)


@st.cache_data(show_spinner=False)
def fixed_cached(
    ticker: str,
    last_date: str,
    key: str,
    name: str,
    kelly_settings: tuple[str, float, float, float],
    params_index: int,
    cost_bps: float,
    slippage_bps: float,
    ppy: float,
    _prices: pd.Series,
    _rf: pd.Series,
) -> BacktestResult:
    strategy = strategy_by_name(name, kelly_settings)
    return fixed_params(strategy, _prices, strategy.grid[params_index], _rf, cost_bps, slippage_bps, ppy)


@st.cache_data(show_spinner=False)
def bootstrap_bands(excess: pd.Series, prior_sd_annual: float) -> tuple[KellyBand, KellyBand]:
    """Bootstrap 5-95% bands for (Gaussian, Shrunk) Kelly on the same resamples."""
    idx = block_bootstrap_indices(len(excess))
    return (
        kelly_band(kelly_distribution(excess, idx)),
        kelly_band(kelly_distribution(excess, idx, prior_sd_annual)),
    )


def portfolio_daily_rf() -> pd.Series | None:
    """Daily T-bill rate over the selected History, or None if it can't be fetched."""
    try:
        return daily_rf(fetch_rf_cached(state.years()))
    except Exception:
        return None


@st.cache_resource(max_entries=512, show_spinner=False)
def figure(name: str, key: tuple[object, ...], _build: Callable[[], go.Figure]) -> go.Figure:
    """
    Memoize a built Plotly figure by (name, key) so warm page switches skip rebuilding it. key must include every
    input the figure depends on. Figures are shared across sessions and must be treated as read-only.
    """
    return _build()
