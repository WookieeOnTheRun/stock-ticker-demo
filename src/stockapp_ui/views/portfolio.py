"""Portfolio page: watchlist input, robust estimation, constrained Kelly vs HRP, frontier and trades."""

from collections.abc import Callable

import pandas as pd
import streamlit as st

from stockapp.charts import correlation_heatmap, dendrogram_chart, frontier_chart, growth_chart, weights_chart
from stockapp.portfolio import CONSTRAINED, UNCONSTRAINED, PortfolioAnalysis, analyze_portfolio
from stockapp.portfolio_data import InsufficientDataError, ReturnsData, default_frequency, returns_matrix
from stockapp.portfolio_input import (
    MAX_TICKERS,
    MAX_UPLOAD_BYTES,
    Watchlist,
    WatchlistError,
    parse_ticker_text,
    parse_watchlist_csv,
    shares_to_weights,
)
from stockapp.trades import TAX_NOTE, trade_plan
from stockapp_ui import cache, state

DEFAULT_TICKERS = "AAPL, MSFT, JNJ, XOM, JPM, SPY"

LoadInBase = Callable[[str, int | None, str], pd.Series]
DailyRf = Callable[[], pd.Series | None]
SectorLookup = Callable[[str], str | None]


@st.cache_data(show_spinner="Estimating and optimizing...")
def _analyze(
    key: str,
    prior: float,
    c: float,
    w_max: float,
    gross_max: float,
    sector_cap: float | None,
    sectors: tuple[str | None, ...],
    _data: ReturnsData,
) -> PortfolioAnalysis:
    """Cached analyze_portfolio: `key` stands for _data (which isn't hashed); the rest are optimizer settings."""
    caps = {s: sector_cap for s in set(sectors) if s} if sector_cap is not None else None
    return analyze_portfolio(_data, prior, c, w_max, gross_max, caps, list(sectors))


def _sectors(tickers: list[str], lookup: SectorLookup) -> tuple[str | None, ...]:
    """Sector per ticker; a failed lookup gives None, so that ticker is simply not sector-capped."""
    found: list[str | None] = []
    for ticker in tickers:
        try:
            found.append(lookup(ticker))
        except Exception:
            found.append(None)
    return tuple(found)


def _watchlist_input() -> Watchlist | None:
    """The typed or uploaded watchlist, or None if there is none yet or it is invalid (the error is shown)."""
    source = st.radio("Watchlist", ["Type tickers", "Upload CSV"], horizontal=True, key="pf_source")
    if source == "Type tickers":
        text = st.text_area(
            f"Tickers (up to {MAX_TICKERS}, separated by commas or spaces)", DEFAULT_TICKERS, key="pf_tickers"
        )
        try:
            return Watchlist(parse_ticker_text(text))
        except WatchlistError as exc:
            st.error(str(exc))
            return None
    upload = st.file_uploader(
        "CSV with a 'ticker' column and optionally 'weight' or 'shares' (max 1 MB, 50 tickers)",
        type=["csv"],
        max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
        key="pf_upload",
    )
    if upload is None:
        st.caption("Example: ticker,weight / MSFT,0.30 / AAPL,0.20 — weights are fractions; the rest is cash.")
        return None
    try:
        return parse_watchlist_csv(upload.getvalue())
    except WatchlistError as exc:
        st.error(str(exc))
        return None


def _risk_free(daily_rf: DailyRf) -> tuple[pd.Series, str] | None:
    """
    Daily T-bill rate, or a constant manual rate entered here if the T-bill fetch failed, with a short label of
    which one is in use (part of the analysis cache key, so changing the manual rate recomputes the analysis).
    """
    rf = daily_rf()
    if rf is not None:
        return rf, "irx"
    st.warning("Couldn't fetch the T-bill rate. Enter a risk-free rate; a manual rate is in use.")
    manual = st.number_input(
        "Annual risk-free rate (%)", min_value=0.0, max_value=20.0, value=None, step=0.1, format="%.2f", key="pf_rf"
    )
    if manual is None:
        return None
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * 60)
    return pd.Series((1 + manual / 100) ** (1 / 252) - 1, index=index), f"manual {manual:g}"


def render() -> None:
    """Render the page: watchlist, prices in the base currency, risk-free rate, frequency, then the analysis."""
    load_in_base: LoadInBase = cache.load_in_base
    daily_rf: DailyRf = cache.portfolio_daily_rf
    sector_lookup: SectorLookup = cache.sector_cached
    years = state.years()
    base_currency = str(st.session_state.get("base_currency", "USD"))
    st.caption(f"Uses the sidebar's base currency ({base_currency}), Kelly fraction and prior SD settings.")
    watchlist = _watchlist_input()
    if watchlist is None or not watchlist.tickers:
        return
    if len(watchlist.tickers) < 2:
        st.info("Enter at least two tickers to build a portfolio.")
        return

    prices: dict[str, pd.Series] = {}
    failed: list[str] = []
    for ticker in watchlist.tickers:
        try:
            prices[ticker] = load_in_base(ticker, years, base_currency)
        except Exception:
            failed.append(ticker)
    if failed:
        st.warning(f"Could not load {', '.join(failed)}; they are left out.")
    if len(prices) < 2:
        st.info("At least two tickers with price data are needed.")
        return

    latest = {t: float(p.iloc[-1]) for t, p in prices.items()}
    current: dict[str, float] = {t: 0.0 for t in prices}
    holdings_value: float | None = None
    if watchlist.weights is not None:
        current.update({t: w for t, w in watchlist.weights.items() if t in prices})
    elif watchlist.shares is not None:
        weights, holdings_value = shares_to_weights(watchlist.shares, latest)
        current.update(weights)
    st.dataframe(
        pd.DataFrame(
            {
                f"Latest price ({base_currency})": latest,
                "Current weight": current,
                "History from": {t: p.index[0].date() for t, p in prices.items()},
            }
        )
    )
    if holdings_value is not None:
        st.caption(f"Holdings value from shares × latest prices: {holdings_value:,.2f} {base_currency}.")

    risk_free = _risk_free(daily_rf)
    if risk_free is None:
        return
    rf, rf_key = risk_free
    frequencies = ["Daily", "Weekly"]
    suggested = default_frequency(list(prices))
    # The key includes the suggestion, so the default follows it when the ticker mix changes (a widget keeps its own
    # value for as long as its key is unchanged).
    frequency = st.selectbox(
        "Return frequency", frequencies, index=frequencies.index(suggested), key=f"pf_freq_{suggested}"
    )
    st.caption(
        "Exchanges close at different times (e.g. New York vs Hong Kong), so their daily returns cover different "
        "windows and daily correlations are biased toward zero. Weekly returns largely remove this; the default is "
        "Weekly when the tickers span different exchanges. Weekly data is annualized with 52 periods."
    )
    try:
        data: ReturnsData = returns_matrix(prices, rf, "Weekly" if frequency == "Weekly" else "Daily")
    except InsufficientDataError as exc:
        st.error(str(exc))
        return
    st.caption(
        f"{data.common_days} trading dates are common to all {len(prices)} tickers "
        f"({len(data.returns)} {frequency.lower()} returns)."
    )

    _render_analysis(
        data, prices, current, latest, holdings_value, base_currency, sector_lookup, years, frequency, rf_key
    )


def _render_analysis(
    data: ReturnsData,
    prices: dict[str, pd.Series],
    current: dict[str, float],
    latest: dict[str, float],
    holdings_value: float | None,
    base_currency: str,
    sector_lookup: SectorLookup,
    years: int | None,
    frequency: str,
    rf_key: str,
) -> None:
    """Optimizer settings, estimates, weights and bands, frontier, clustering, split-sample test and trade list."""
    tickers = [str(t) for t in data.returns.columns]
    c = float(st.session_state.get("k_fraction", 0.5))
    prior = float(st.session_state.get("prior_sd_annual", 0.10))
    k1, k2, k3, k4 = st.columns(4)
    w_max = k1.slider("Max weight per asset", 0.05, 1.0, 0.25, 0.05, key="pf_wmax")
    gross_max = k2.slider("Max total invested (rest is cash)", 0.1, 1.0, 1.0, 0.05, key="pf_gross")
    cost_bps = k3.number_input("Transaction cost (bps)", 0.0, 100.0, 0.0, 0.5, key="pf_cost")
    value_input = k4.number_input(
        f"Portfolio value ({base_currency}, for share counts)", 0.0, 1e12, holdings_value or 0.0, 1000.0, key="pf_value"
    )
    use_sector_caps = st.checkbox("Cap each sector (yfinance sectors)", key="pf_sector_on")
    sector_cap = (
        st.slider("Max weight per sector", 0.05, 1.0, 0.40, 0.05, key="pf_sector_cap") if use_sector_caps else None
    )
    sectors = _sectors(tickers, sector_lookup) if use_sector_caps else tuple(None for _ in tickers)
    if use_sector_caps and all(s is None for s in sectors):
        st.info("No sectors could be looked up, so no sector caps apply.")
    st.caption(
        f"Kelly fraction c = {c:g} and prior SD {prior:.0%} come from the sidebar. Long-only: no shorting or leverage."
    )

    # Cache key for _analyze: everything `data` was built from (the data itself is not hashed).
    key = "|".join([*tickers, str(data.returns.index[-1].date()), base_currency, str(years), frequency, rf_key])
    analysis = _analyze(key, prior, c, w_max, gross_max, sector_cap, sectors, data)
    est = analysis.estimates
    cov = est.covariance

    e1, e2, e3 = st.columns(3)
    e1.metric("Ledoit–Wolf shrinkage", f"{cov.shrinkage:.3f}")
    e2.metric(
        "Condition number", f"{cov.condition_shrunk:,.0f}", f"sample {cov.condition_sample:,.0f}", delta_color="off"
    )
    e3.metric("Observations", f"{len(data.returns)} {frequency.lower()}")
    if cov.jitter > 0:
        st.caption(f"The covariance was not positive definite; {cov.jitter:.2e} was added to its diagonal.")
    st.caption("Means are shrunk toward 0 like the Research page's Shrunk Kelly; covariance shrinkage stabilizes Σ⁻¹.")

    allocation = analysis.allocation
    if allocation.failed:
        st.warning(
            "The constrained optimizer failed, so there is no Constrained Kelly recommendation and no trades are "
            f"proposed below. The other methods are unaffected. Reason: {allocation.reason}"
        )
    elif allocation.status != "optimal":
        st.caption(
            f"The solver reported '{allocation.status}'; the solution was checked against every constraint before use."
        )

    table = analysis.weights.copy()
    table.loc["Cash"] = 1 - table.sum(skipna=False)  # a failed (NaN) column stays NaN rather than reading as all cash
    table["Constrained 5%"] = analysis.bands.p5.reindex(table.index)
    table["Constrained 95%"] = analysis.bands.p95.reindex(table.index)
    st.dataframe(table.style.format("{:.1%}", na_rep="—"))
    st.caption(
        f"{UNCONSTRAINED}: c·Σ⁻¹μ — unconstrained: may include shorts and leverage, shown for reference only. "
        f"Bands: {analysis.bands.solved} block-bootstrap resamples (5th–95th percentile)."
    )
    if analysis.bands.reduced:
        st.info(
            f"Bootstrap reduced to {analysis.bands.resamples} of {analysis.bands.requested} resamples to stay within "
            "the time budget for this many tickers; the bands are a little noisier."
        )
    if analysis.bands.failures:
        st.caption(
            f"The optimizer failed on {analysis.bands.failures} of {analysis.bands.resamples} resamples; they are "
            "left out of the bands."
        )

    # A failed Constrained Kelly column is all NaN; dropping it leaves the chart to the methods that have weights.
    chart_weights = analysis.weights.dropna(axis=1, how="all")
    st.plotly_chart(weights_chart(chart_weights, analysis.bands.p5, analysis.bands.p95, banded=CONSTRAINED))
    st.caption("How to read this: bars are weights per method; error bars on Constrained Kelly are its bootstrap band.")
    st.plotly_chart(frontier_chart(analysis.frontier, analysis.marks))
    st.caption(
        "How to read this: each frontier point is the best long-only portfolio for one Kelly fraction; points further "
        "up and left are better. Expected returns are the shrunk estimates, not forecasts."
    )
    corr = data.returns.corr().loc[analysis.hrp.order, analysis.hrp.order]
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(correlation_heatmap(corr))
    with h2:
        st.plotly_chart(dendrogram_chart(analysis.hrp.linkage, tickers))
    st.caption("How to read this: blocks of red mark assets that move together; the tree shows how HRP groups them.")
    st.plotly_chart(growth_chart(analysis.split_growth))
    st.caption(
        f"In-sample / out-of-sample split: weights estimated on data before {analysis.split_date.date()} and held "
        "fixed (rebalanced each period) afterwards. A full walk-forward version is on the Backtest & Signals page."
    )
    if analysis.split_failure:
        st.caption(
            "Constrained Kelly is not in this chart: the optimizer failed on the first half "
            f"({analysis.split_failure})."
        )

    st.markdown("**Current vs target (Constrained Kelly)**")
    if allocation.failed:
        st.info("No trades proposed: the constrained optimizer failed, so there is no target to trade toward.")
        return
    target = analysis.weights[CONSTRAINED].to_dict()
    plan = trade_plan(current, target, cost_bps, latest, value_input or None)
    formats = {c: "{:.1%}" for c in plan.table.columns if c != "Trade (shares)"}
    formats["Trade (shares)"] = "{:,.1f}"
    st.dataframe(plan.table.style.format(formats, na_rep="—"))
    t1, t2 = st.columns(2)
    t1.metric("Turnover", f"{plan.turnover:.1%}")
    cost_text = f"{plan.cost_fraction:.3%} of portfolio"
    if value_input:
        cost_text += f" ≈ {plan.cost_fraction * value_input:,.2f} {base_currency}"
    t2.metric("Transaction cost", cost_text)
    st.caption(f"Turnover = ½·Σ|Δw| including cash. {TAX_NOTE}")
