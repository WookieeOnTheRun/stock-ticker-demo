"""Research page: one ticker's prices, Kelly sizing, and risk, distribution, regime, factor and relative analytics."""

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stockapp import factors, regimes, relative, risk, seasonality
from stockapp.calendars import calendar_code
from stockapp.charts import (
    acf_chart,
    cumulative_return_chart,
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
from stockapp.kelly import excess_returns
from stockapp.metrics import annualized_volatility, cagr, growth_rate, sharpe_ratio
from stockapp_ui import cache
from stockapp_ui.cache import (
    bootstrap_bands,
    distribution_cached,
    factor_fit_cached,
    factors_cached,
    garch_cached,
    hmm_cached,
    load_actions_cached,
    load_in_base,
    regimes_cached,
    risk_cached,
    sector_cached,
)
from stockapp_ui.context import build_context
from stockapp_ui.state import METHODS

WINDOW_SIZES = [60, 120, 252, 504]
BOOTSTRAP_METHODS = {"Gaussian", "Shrunk"}
MIN_REGIME_DAYS = 60
MIN_FACTOR_DAYS = 60


def render() -> None:
    ctx = build_context()
    k_fraction_key, max_dd_pct_key = ctx.k_fraction, ctx.max_dd_pct
    ticker = ctx.ticker
    close = ctx.close
    close_all = ctx.close_all
    returns_df = ctx.returns_df
    returns_all = ctx.returns_all
    rf_df = ctx.rf_df
    rf_all = ctx.rf_all
    excess = ctx.excess
    kr = ctx.kr
    raws = ctx.raws
    history_label = ctx.history_label
    years = ctx.years
    base_currency = ctx.base_currency
    price_currency = ctx.price_currency
    native_currency = ctx.native_currency
    method = ctx.method
    prior_sd_annual = ctx.prior_sd_annual
    max_dd_pct = ctx.max_dd_pct
    periods_per_year = ctx.periods_per_year
    gauss_band = ctx.gauss_band
    shrunk_band = ctx.shrunk_band
    method_bands = ctx.method_bands
    applied = ctx.applied
    benchmark = ctx.benchmark
    benchmark_close = ctx.benchmark_close
    benchmark_problem = ctx.benchmark_problem
    last_date = ctx.last_date
    settings_key = ctx.settings_key
    r_look = returns_df["simple_return"]
    r_all = returns_all["simple_return"]
    applied_f = applied(raws[method])
    # Every input any chart on this page depends on; each chart is memoized under (its name, this key).
    page_key = (ticker, last_date, ctx.settings_key, method, k_fraction_key, prior_sd_annual, max_dd_pct_key, benchmark)

    def chart(name: str, build: Callable[[], go.Figure], *extra: object) -> None:
        st.plotly_chart(cache.figure(name, (*page_key, *extra), build))

    # Charts
    st.subheader("Charts")
    tab_price, tab_returns, tab_risk, tab_dist, tab_regimes, tab_factors, tab_relative = st.tabs(
        ["Price", "Returns", "Risk", "Distribution", "Regimes", "Factors", "Relative"]
    )
    with tab_price:
        chart("c2", lambda: price_chart(close, load_actions_cached(ticker), dividend_currency=native_currency or ""))
        st.caption(
            "How to read this: the line is the adjusted close; dashed lines mark splits and dots mark dividends."
        )
    with tab_returns:
        chart("c3", lambda: returns_histogram(returns_df))
        st.caption("How to read this: each bar counts days whose return fell in that range; wide tails mean big days.")
        chart("c4", lambda: cumulative_return_chart(returns_df))
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
            f"days. Max drawdown: peak {dd.peak_date.date()}, trough {dd.trough_date.date()}, recovery {recovery} "
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
        chart("c5", lambda: volatility_chart(rk.rolling_vol, rk.ewma_vol))
        st.caption("How to read this: higher lines mean bigger daily swings; EWMA reacts faster to recent shocks.")
        chart("c6", lambda: underwater_chart(close))
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
            chart("c7", lambda: qq_chart(dist.qq_normal, "Q-Q vs Normal (standardized)"))
        with q2:
            chart("c8", lambda: qq_chart(dist.qq_t, f"Q-Q vs Student-t (fitted df {dist.t_fit.df:.1f})"))
        st.caption(
            "How to read this: points on the dashed line mean the model fits; ends bending away from it mean the real "
            "tails are fatter than the model's."
        )
        chart("c9", lambda: acf_chart(dist.acf, dist.acf_squared, dist.acf_band))
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
                "trading days. Choose a longer History."
            )
        else:
            current = labels.dropna().iloc[-1]
            st.metric("Current volatility regime", str(current))
            chart("c10", lambda: regime_chart(close_all.loc[labels.index], labels, "Price by Volatility Regime"))
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
                chart(
                    "c11",
                    lambda: regime_chart(
                        close_all.loc[hmm_labels.index], hmm_labels, "HMM Regimes — historical view (uses full sample)"
                    ),
                    n_states,
                )
                st.caption(
                    "Historical view (uses full sample): each label is decoded with knowledge of later data, so it "
                    "looks ahead. It is for context only and is never used for sizing or signals."
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
                chart("c12", lambda: factor_chart(loadings, ffit.std_errors.drop("alpha"), factors.FACTOR_LABELS))
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
            chart("c13", lambda: growth_chart(relative.growth_of_one({ticker: close_all, **comparisons})))
            st.caption("How to read this: what $1 grew to in each, from the first date they all have prices.")
            table = relative.relative_table(close_all, comparisons).rename(columns={"stock": ticker})
            st.dataframe(table.style.format("{:.1%}"))
            st.caption(
                "Trailing returns over 1, 3, 6 and 12 months (21/63/126/252 trading days) and the stock's excess."
            )
            chart("c14", lambda: relative_chart(relative.rolling_relative(close_all, benchmark_close), str(benchmark)))
            st.caption(f"How to read this: above zero, {ticker} beat {benchmark} over that trailing window.")
        st.markdown("**Seasonality**")
        monthly = seasonality.monthly_returns(close_all)
        chart("c15", lambda: seasonality_heatmap(monthly))
        chart("c16", lambda: month_average_chart(seasonality.average_by_month(monthly)))
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
        kelly_by_window[w] = applied(ctx.raw_kelly(method, r_w, rf_w))
        if method in BOOTSTRAP_METHODS:
            g_band, s_band = bootstrap_bands(excess_returns(r_w, rf_w), prior_sd_annual)
            band = g_band if method == "Gaussian" else s_band
            window_bands[w] = (applied(band.p5), applied(band.p95))
    chart(
        "c17",
        lambda: kelly_over_windows_chart(
            kelly_by_window, bands=window_bands, title=f"Applied Kelly Across Lookback Windows ({method})"
        ),
    )
    if method in BOOTSTRAP_METHODS:
        st.caption("Error bars: bootstrap 5–95% range of the applied fraction.")

    # Data preview and download
    st.subheader("Data")
    export_df = returns_df.assign(rf_daily=rf_df, excess_return=excess)
    st.dataframe(export_df.tail(20))
    csv = export_df.to_csv().encode("utf-8")
    st.download_button("Download returns CSV", data=csv, file_name=f"{ticker}_returns.csv", mime="text/csv")
