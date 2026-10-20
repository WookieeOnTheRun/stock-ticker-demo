from datetime import date

import pandas as pd
import streamlit as st

import portfolio_tab
from stockapp import distribution, factors, regimes, relative, risk, seasonality
from stockapp.backtest.engine import DEFAULT_COST_BPS, BacktestResult
from stockapp.backtest.strategies import BUY_AND_HOLD, SIGNALS, Strategy, kelly_buy_and_hold
from stockapp.backtest.walkforward import TRAIN_YEARS, TRIALS_CAPTION, WalkForwardResult, fixed_params, walk_forward
from stockapp.bootstrap import KellyBand, block_bootstrap_indices, kelly_band, kelly_distribution
from stockapp.calendars import calendar_code, is_stale, sessions_per_year
from stockapp.charts import (
    acf_chart,
    cumulative_return_chart,
    drawdowns_chart,
    factor_chart,
    growth_chart,
    kelly_over_windows_chart,
    month_average_chart,
    price_chart,
    qq_chart,
    regime_chart,
    relative_chart,
    returns_histogram,
    seasonality_heatmap,
    underwater_chart,
    volatility_chart,
)
from stockapp.data import compute_returns, normalize_ticker
from stockapp.drawdown import drawdown_constrained_kelly
from stockapp.fx import BASE_CURRENCIES, convert_prices, fx_ticker, major_currency
from stockapp.kelly import excess_returns, fractional_kelly, log_optimal_kelly, shrunk_kelly
from stockapp.metrics import annualized_volatility, cagr, growth_rate, sharpe_ratio
from stockapp.providers.yfinance_provider import YFinanceProvider
from stockapp.repository import load_actions, load_bars, load_currency
from stockapp.riskfree import align_rf, daily_rf, fetch_rf_annual_pct

WINDOW_SIZES = [60, 120, 252, 504]
METHODS = ["Gaussian", "Shrunk", "Log-optimal", "Drawdown-constrained"]
SURVIVORSHIP_NOTE = (
    "Analyses use currently listed tickers; delisted companies are absent, which biases historical comparisons upward."
)
HISTORY_OPTIONS: dict[str, int | None] = {"5y": 5, "10y": 10, "20y": 20, "max": None}
BOOTSTRAP_METHODS = {"Gaussian", "Shrunk"}
BENCHMARKS = ["SPY", "QQQ", "IWM", "DIA", "Other"]
MIN_REGIME_DAYS = 60
MIN_FACTOR_DAYS = 60

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


st.set_page_config(page_title="Fractional Kelly Analyzer", layout="wide")

st.title("📈 Fractional Kelly Stock Analyzer")


def portfolio_daily_rf() -> pd.Series | None:
    """Daily T-bill rate for the Portfolio tab over the Stock tab's History, or None if it can't be fetched."""
    try:
        return daily_rf(fetch_rf_cached(HISTORY_OPTIONS[str(st.session_state.get("history", "10y"))]))
    except Exception:
        return None


# The Portfolio tab is rendered first: the Stock tab below may st.stop() on bad input.
tab_stock, tab_portfolio = st.tabs(["Stock", "Portfolio"])
with tab_portfolio:
    portfolio_tab.render(load_in_base, portfolio_daily_rf, HISTORY_OPTIONS[str(st.session_state.get("history", "10y"))])

with tab_stock:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        raw_ticker = st.text_input(
            "Ticker",
            value="MSFT",
            placeholder="e.g., AAPL, MSFT, SPY, VOD.L",
            help=SURVIVORSHIP_NOTE,
        )
        history_label = st.selectbox(
            "History", list(HISTORY_OPTIONS), index=list(HISTORY_OPTIONS).index("10y"), key="history"
        )
        base_currency = st.selectbox(
            "Base currency", BASE_CURRENCIES, index=BASE_CURRENCIES.index("USD"), key="base_currency"
        )
        benchmark_choice = st.selectbox("Benchmark", BENCHMARKS, index=0)
        raw_benchmark = (
            st.text_input("Benchmark ticker", value="SPY", placeholder="e.g., VT, EFA, ^GSPC")
            if benchmark_choice == "Other"
            else benchmark_choice
        )
        lookback_slot = st.empty()  # filled once the available history is known
        k_fraction = st.slider(
            "Kelly fraction multiplier", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="k_fraction"
        )
        prior_sd_annual = st.number_input(
            "Prior SD of annual expected excess return",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            format="%.2f",
            key="prior_sd_annual",
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
        native_close, native_currency = load_close(ticker, HISTORY_OPTIONS[history_label])
    except Exception as e:
        st.error(f"Could not load data for {ticker}: {e}")
        st.stop()

    # Convert to the base currency (minor units such as GBp -> GBP first). Returns, and so Kelly, are measured in it.
    major, divisor = major_currency(native_currency or base_currency)
    pair = None if major == base_currency else fx_ticker(major, base_currency)
    try:
        fx_rate = load_fx(pair, HISTORY_OPTIONS[history_label]) if pair else None
        close_all = convert_prices(native_close, divisor, fx_rate)
        price_currency = base_currency
    except Exception:
        st.warning(f"Couldn't load the {pair} FX rate; prices are shown in {major}, not converted to {base_currency}.")
        close_all = convert_prices(native_close, divisor, None)
        price_currency = major
    if close_all.empty:
        st.error(f"No {pair} FX rates overlap {ticker}'s price history.")
        st.stop()
    with col_left:
        if native_currency is None:
            st.caption(f"Quote currency unknown; prices assumed to be in {base_currency}.")
        elif native_currency == price_currency:
            st.caption(f"Prices in {price_currency}.")
        else:
            st.caption(f"Prices in {native_currency} (native), converted to {price_currency}.")

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

    # Analytics inputs. Risk and distribution use the lookback window (as Kelly does); regimes, factors, relative
    # strength
    # and seasonality use the full loaded history because they need long samples.
    r_look = returns_df["simple_return"]
    r_all = returns_all["simple_return"]
    last_date = str(close_all.index[-1].date())
    settings_key = f"{base_currency}|{history_label}|{lookback}|{manual_rf_pct if manual_rf_pct is not None else 'irx'}"
    years = HISTORY_OPTIONS[history_label]

    benchmark = normalize_ticker(raw_benchmark or "")
    benchmark_close: pd.Series | None = None
    benchmark_problem = ""
    if benchmark is None:
        benchmark_problem = "Invalid benchmark ticker. Use letters, digits, and . ^ = - only (max 15 characters)."
    elif benchmark == ticker:
        benchmark_problem = f"The benchmark is {ticker} itself; choose a different benchmark."
    else:
        try:
            benchmark_close = load_in_base(benchmark, years, base_currency)
        except Exception as e:
            benchmark_problem = f"Could not load benchmark {benchmark}: {e}"

    # Charts
    st.subheader("Charts")
    tab_price, tab_returns, tab_risk, tab_dist, tab_regimes, tab_factors, tab_relative, tab_backtest = st.tabs(
        ["Price", "Returns", "Risk", "Distribution", "Regimes", "Factors", "Relative", "Backtest"]
    )
    with tab_price:
        st.plotly_chart(price_chart(close, load_actions_cached(ticker), dividend_currency=native_currency or ""))
        st.caption(
            "How to read this: the line is the adjusted close; dashed lines mark splits and dots mark dividends."
        )
    with tab_returns:
        st.plotly_chart(returns_histogram(returns_df))
        st.caption("How to read this: each bar counts days whose return fell in that range; wide tails mean big days.")
        st.plotly_chart(cumulative_return_chart(returns_df))
        st.caption("How to read this: the value of $1 invested at the start of the lookback window.")

    with tab_risk:
        st.caption(f"Computed over the {len(r_look)}-day lookback window, in {price_currency}.")
        rk = risk_cached(ticker, last_date, settings_key, periods_per_year, r_look, close, rf_df)
        dd = rk.drawdown
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1-day VaR 95% (historical)", f"{rk.historical_var:.2%}")
        c2.metric("1-day CVaR 95% (historical)", f"{rk.historical_cvar:.2%}")
        c3.metric("1-day VaR 95% (normal)", f"{rk.parametric_var:.2%}")
        c4.metric("Max drawdown", f"{-dd.depth:.1%}")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Sortino", f"{rk.sortino:.2f}")
        c6.metric("Calmar", f"{rk.calmar:.2f}")
        c7.metric("Omega (threshold 0)", f"{rk.omega:.2f}")
        if benchmark_close is not None:
            fit = risk.beta_r2(r_look, benchmark_close.pct_change().dropna(), periods_per_year)
            c8.metric(f"Beta vs {benchmark}", f"{fit.beta:.2f}", f"R² {fit.r2:.2f}, alpha {fit.alpha_annual:.1%}/yr")
        else:
            c8.metric("Beta", "—")
            st.info(f"{benchmark_problem} Beta is unavailable.")
        recovery = dd.recovery_date.date() if dd.recovery_date is not None else "not yet recovered"
        st.caption(
            "VaR/CVaR are losses: on 1 day in 20 expect to lose at least the VaR; CVaR is the average loss on those "
            "days. "
            f"Max drawdown: peak {dd.peak_date.date()}, trough {dd.trough_date.date()}, recovery {recovery} "
            f"({dd.duration_days} calendar days)."
        )
        garch = garch_cached(ticker, last_date, settings_key, periods_per_year, r_look)
        ewma_now = float(rk.ewma_vol.iloc[-1])
        if garch is not None and garch.converged:
            st.metric(
                "Forecast volatility, next 21 days (GARCH(1,1))",
                f"{garch.annualized_vol:.1%}",
                f"EWMA now {ewma_now:.1%}",
            )
        else:
            st.metric("Forward volatility (EWMA, λ = 0.94)", f"{ewma_now:.1%}")
            st.caption("GARCH(1,1) did not converge on this window, so EWMA is used as the forward estimate.")
        st.plotly_chart(volatility_chart(rk.rolling_vol, rk.ewma_vol))
        st.caption("How to read this: higher lines mean bigger daily swings; EWMA reacts faster to recent shocks.")
        st.plotly_chart(underwater_chart(close))
        st.caption("How to read this: how far the price sat below its previous high; 0% means at a new high.")

    with tab_dist:
        st.caption(f"Computed over the {len(r_look)}-day lookback window.")
        dist = distribution_cached(ticker, last_date, settings_key, r_look)
        verdict = dist.verdict
        st.markdown("**Is Gaussian Kelly safe for this stock?**")
        gap = raws["Log-optimal"] - raws["Gaussian"]
        gap_text = (
            f" Log-optimal Kelly is {raws['Log-optimal']:.2f} vs Gaussian {raws['Gaussian']:.2f} (gap {gap:+.2f}); "
            "a large negative gap is the fat-tail penalty."
        )
        (st.warning if (verdict.fat_tails or verdict.non_normal or verdict.volatility_clustering) else st.success)(
            verdict.message + gap_text
        )
        jb_stat, jb_p = dist.jarque_bera
        lb_q, lb_p = dist.ljung_box
        lb2_q, lb2_p = dist.ljung_box_squared
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Skewness", f"{dist.skewness:.2f}")
        d2.metric("Excess kurtosis", f"{verdict.excess_kurtosis:.2f}")
        d3.metric("Jarque–Bera p", f"{jb_p:.3g}", f"JB {jb_stat:.0f}")
        d4.metric("Ljung–Box(10) p, returns", f"{lb_p:.3g}", f"Q {lb_q:.1f}")
        d5.metric("Ljung–Box(10) p, squared", f"{lb2_p:.3g}", f"Q {lb2_q:.1f}")
        st.metric("Hurst exponent (indicative; noisy on short samples)", f"{dist.hurst:.2f}")
        st.caption(
            "Hurst ≈ 0.5 is a random walk, > 0.5 trending, < 0.5 mean-reverting; R/S is biased up on short samples."
        )
        q1, q2 = st.columns(2)
        with q1:
            st.plotly_chart(qq_chart(dist.qq_normal, "Q-Q vs Normal (standardized)"))
        with q2:
            st.plotly_chart(qq_chart(dist.qq_t, f"Q-Q vs Student-t (fitted df {dist.t_fit.df:.1f})"))
        st.caption(
            "How to read this: points on the dashed line mean the model fits; ends bending away from it mean the real "
            "tails are fatter than the model's."
        )
        st.plotly_chart(acf_chart(dist.acf, dist.acf_squared, dist.acf_band))
        st.caption(
            "How to read this: bars outside the dashed ±1.96/√n lines are significant; tall squared-return bars mean "
            "volatility clusters."
        )

    with tab_regimes:
        st.caption(f"Computed over the full {history_label} history ({len(r_all)} days).")
        labels, regime_kelly = regimes_cached(ticker, last_date, settings_key, prior_sd_annual, r_all, rf_all)
        if labels.dropna().empty:
            st.info(
                f"Not enough history for volatility regimes: they need {regimes.VOL_WINDOW + regimes.MIN_HISTORY} "
                "trading "
                "days. Choose a longer History."
            )
        else:
            current = labels.dropna().iloc[-1]
            st.metric("Current volatility regime", str(current))
            st.plotly_chart(regime_chart(close_all.loc[labels.index], labels, "Price by Volatility Regime"))
            st.caption(
                "How to read this: shading shows each day's 21-day volatility against all earlier days (bottom third = "
                "low, top third = high). Labels use no future data."
            )
            st.dataframe(
                regime_kelly.rename(
                    columns={"days": "Days", "gaussian_kelly": "Gaussian f*", "shrunk_kelly": "Shrunk f*"}
                )
            )
            thin = regime_kelly.index[regime_kelly["days"] < MIN_REGIME_DAYS].tolist()
            if thin:
                st.warning(
                    f"Fewer than {MIN_REGIME_DAYS} days in: {', '.join(thin)}. Those Kelly estimates are unreliable."
                )
            st.caption("Raw (unclamped) Kelly using only the days in each regime; excess over the T-bill rate.")
        if st.toggle("Show HMM regimes (historical view, uses full sample)"):
            n_states = st.radio("HMM states", [2, 3], horizontal=True)
            hmm_labels = hmm_cached(ticker, last_date, settings_key, int(n_states), r_all)
            if hmm_labels is None:
                st.info("The HMM could not be fitted on this history.")
            else:
                st.plotly_chart(
                    regime_chart(
                        close_all.loc[hmm_labels.index], hmm_labels, "HMM Regimes — historical view (uses full sample)"
                    )
                )
                st.caption(
                    "Historical view (uses full sample): each label is decoded with knowledge of later data, so it "
                    "looks "
                    "ahead. It is for context only and is never used for sizing or signals."
                )

    with tab_factors:
        st.caption("US Fama–French 5 factors + momentum. The regression is less meaningful for non-US tickers.")
        try:
            factor_data = factors_cached()
        except Exception as e:
            factor_data = None
            st.info(f"Factor data is unavailable right now ({e}).")
        if factor_data is not None:
            through = str(factor_data.index[-1].date())
            st.caption(f"Factor data through {through} (the library lags by several weeks).")
            overlap = factors.stock_excess_returns(r_all, factor_data)
            if len(overlap) < MIN_FACTOR_DAYS:
                st.info(f"Only {len(overlap)} days overlap the factor data; at least {MIN_FACTOR_DAYS} are needed.")
            else:
                ffit = factor_fit_cached(ticker, last_date, settings_key, through, r_all, factor_data)
                if calendar_code(ticker) != "XNYS":
                    st.info(
                        "This is not a US listing: US factors explain non-US stocks poorly, so read the fit loosely."
                    )
                st.markdown(f"**{factors.describe_fit(ffit)}**")
                f1, f2, f3 = st.columns(3)
                f1.metric("R²", f"{ffit.r2:.2f}")
                f2.metric(
                    "Alpha (annualized)",
                    f"{ffit.coefs['alpha'] * periods_per_year:.1%}",
                    f"t {ffit.t_stats['alpha']:.1f}",
                )
                f3.metric("Days in regression", f"{ffit.n_obs}")
                loadings = ffit.coefs.drop("alpha")
                st.plotly_chart(factor_chart(loadings, ffit.std_errors.drop("alpha"), factors.FACTOR_LABELS))
                st.caption(
                    "How to read this: a bar is the stock's sensitivity to that factor; if its ±2 SE bar crosses "
                    "zero the "
                    "loading is not statistically significant (Newey–West standard errors)."
                )

    with tab_relative:
        st.caption(f"Over the full {history_label} history, in {price_currency}.")
        if benchmark_close is None:
            st.info(f"{benchmark_problem} Relative performance is unavailable.")
        else:
            sector_lookup_failed = False
            try:
                sector = sector_cached(ticker)
            except Exception:
                sector, sector_lookup_failed = None, True
            etf = relative.sector_etf(sector)
            comparisons: dict[str, pd.Series] = {str(benchmark): benchmark_close}
            if etf is not None and etf not in (ticker, benchmark):
                try:
                    comparisons[etf] = load_in_base(etf, years, base_currency)
                except Exception:
                    st.info(f"Could not load sector ETF {etf}; showing the benchmark only.")
            if sector_lookup_failed:
                st.caption(
                    "No sector comparison: the sector lookup is unavailable right now (offline or provider error)."
                )
            elif etf is None:
                st.caption(
                    f"No sector comparison: yfinance reports no mapped sector for {ticker} ({sector or 'none'})."
                )
            else:
                st.caption(f"Sector: {sector} → {etf}.")
            st.plotly_chart(growth_chart(relative.growth_of_one({ticker: close_all, **comparisons})))
            st.caption("How to read this: what $1 grew to in each, from the first date they all have prices.")
            table = relative.relative_table(close_all, comparisons).rename(columns={"stock": ticker})
            st.dataframe(table.style.format("{:.1%}"))
            st.caption(
                "Trailing returns over 1, 3, 6 and 12 months (21/63/126/252 trading days) and the stock's excess."
            )
            st.plotly_chart(relative_chart(relative.rolling_relative(close_all, benchmark_close), str(benchmark)))
            st.caption(f"How to read this: above zero, {ticker} beat {benchmark} over that trailing window.")
        st.markdown("**Seasonality**")
        monthly = seasonality.monthly_returns(close_all)
        st.plotly_chart(seasonality_heatmap(monthly))
        st.plotly_chart(month_average_chart(seasonality.average_by_month(monthly)))
        st.caption("Seasonal patterns in single stocks are mostly noise; shown for context.")

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

    st.metric(
        label=f"Applied Fractional Kelly ({method})", value=f"{applied_f:.4f}", delta=f"fraction={kr.fraction:.2f}"
    )

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

    with tab_backtest:
        st.caption(f"Over the full {history_label} history, in {price_currency}, with the T-bill rate earned on cash.")
        kelly_settings = (method, float(k_fraction), float(prior_sd_annual), max_dd)
        strategy_names = [*SIGNALS, "Kelly buy & hold"]
        b1, b2, b3, b4, b5 = st.columns(5)
        bt_name = b1.selectbox("Strategy", strategy_names)
        bt_mode = b2.radio("Parameters", ["walk-forward", "fixed"], horizontal=True)
        bt_window = b3.radio("Training window", ["expanding", "rolling"], horizontal=True)
        bt_cost = b4.number_input(
            "Cost (bps per side)", min_value=0.0, max_value=100.0, value=DEFAULT_COST_BPS, step=0.5
        )
        bt_slip = b5.number_input("Slippage (bps per side)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
        st.caption(
            f"Assumptions: {bt_cost:g} bps cost + {bt_slip:g} bps slippage on every change in position; decisions at a "
            f"close earn the next day's return. Walk-forward: train on ≥ {TRAIN_YEARS} years, test the next year, "
            "repeat."
        )
        bt_prices = close_all.loc[rf_all.index[0] :] if len(rf_all) else close_all
        if bt_prices.index[-1] < bt_prices.index[0] + pd.DateOffset(years=TRAIN_YEARS, days=30):
            st.info(f"Backtests need more than {TRAIN_YEARS} years of history; choose a longer History.")
        else:
            # Backtests use the full history, so the lookback slider is not part of their cache key, and only the
            # Kelly strategy depends on the Kelly settings.
            bt_key = f"{base_currency}|{history_label}|{manual_rf_pct if manual_rf_pct is not None else 'irx'}"
            bt_args = (ticker, last_date, bt_key)
            no_kelly = ("", 0.0, 0.0, 0.0)

            def run_wf(name: str) -> WalkForwardResult:
                settings = kelly_settings if name == "Kelly buy & hold" else no_kelly
                return walk_forward_cached(
                    *bt_args, name, settings, bt_cost, bt_slip, bt_window, periods_per_year, bt_prices, rf_all
                )

            bh = run_wf(BUY_AND_HOLD.name)
            kbh = run_wf("Kelly buy & hold")
            strategy = strategy_by_name(bt_name, kelly_settings)
            span_start = bh.positions.index[0]
            columns: dict[str, dict[str, float]] = {}
            curves: dict[str, pd.Series] = {}
            if bt_mode == "walk-forward":
                wf = run_wf(bt_name)
                columns[f"{bt_name} — in-sample"] = wf.in_sample.metrics
                columns[f"{bt_name} — out-of-sample"] = wf.out_of_sample.metrics
                curves[f"{bt_name} (out-of-sample)"] = wf.out_of_sample.equity
                trials = wf.trials + bh.trials + kbh.trials
            else:
                labels = [", ".join(f"{k}={v}" for k, v in p.items()) or "no parameters" for p in strategy.grid]
                params_index = labels.index(st.selectbox("Fixed parameters", labels))
                fixed = fixed_cached(
                    *bt_args,
                    bt_name,
                    kelly_settings if bt_name == "Kelly buy & hold" else no_kelly,
                    params_index,
                    bt_cost,
                    bt_slip,
                    periods_per_year,
                    bt_prices.loc[span_start:],
                    rf_all,
                )
                columns[f"{bt_name} — fixed parameters (in-sample)"] = fixed.metrics
                curves[f"{bt_name} (fixed parameters)"] = fixed.equity
                trials = 1 + bh.trials + kbh.trials
                st.warning(
                    "Fixed parameters are not out-of-sample: they may have been chosen with hindsight, and any "
                    "estimated "
                    "quantity (such as a Kelly fraction) is fitted on the same period it is evaluated on."
                )
            columns["Buy & hold — out-of-sample span"] = bh.out_of_sample.metrics
            columns["Kelly buy & hold — out-of-sample"] = kbh.out_of_sample.metrics
            curves["Buy & hold"] = bh.out_of_sample.equity
            curves["Kelly buy & hold (out-of-sample)"] = kbh.out_of_sample.equity

            equity = pd.DataFrame(curves)
            st.plotly_chart(growth_chart(equity))
            st.caption(
                f"How to read this: growth of $1 from {span_start.date()}, the first out-of-sample decision; every "
                "line "
                "starts on the same day so they can be compared directly."
            )
            st.metric("Trials (parameter combinations evaluated)", trials)
            st.caption(TRIALS_CAPTION)
            table = pd.DataFrame(columns)
            pct_rows = ["CAGR", "Ann. vol", "Max drawdown", "Hit rate", "Exposure"]
            formatted = table.apply(
                lambda row: row.map(lambda v: f"{v:.1%}" if row.name in pct_rows else f"{v:.2f}"), axis=1
            )
            st.dataframe(formatted)
            st.caption(
                "In-sample is the final fold's chosen parameters on the training data that chose them; out-of-sample "
                "stitches each test year, traded with parameters chosen before it. Sharpe and Sortino are on returns "
                "in "
                "excess of the T-bill rate."
            )
            st.plotly_chart(drawdowns_chart(equity))
            st.caption("How to read this: how far each strategy sat below its own previous high.")
