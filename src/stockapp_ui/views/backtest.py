"""Backtest & Signals page: walk-forward backtests and signal cards, the probability model and the advice rules."""

import pandas as pd
import streamlit as st

from stockapp.advice import BANNER, AdviceInputs, advice_probability, causal_skill_gate, decide
from stockapp.backtest.cards import signal_cards
from stockapp.backtest.engine import DEFAULT_COST_BPS
from stockapp.backtest.strategies import BUY_AND_HOLD
from stockapp.backtest.walkforward import TRAIN_YEARS, TRIALS_CAPTION, WalkForwardResult
from stockapp.charts import contributions_chart, drawdowns_chart, fan_chart, growth_chart, reliability_chart
from stockapp.signals import SIGNALS
from stockapp_ui.cache import (
    fan_cached,
    fixed_cached,
    model_cached,
    regimes_cached,
    rule_track_record_cached,
    strategy_by_name,
    walk_forward_cached,
)
from stockapp_ui.context import build_context


def render() -> None:
    ctx = build_context()
    ticker = ctx.ticker
    close_all = ctx.close_all
    returns_all = ctx.returns_all
    rf_all = ctx.rf_all
    raws = ctx.raws
    history_label = ctx.history_label
    price_currency = ctx.price_currency
    method = ctx.method
    k_fraction = ctx.k_fraction
    prior_sd_annual = ctx.prior_sd_annual
    max_dd = ctx.max_dd
    periods_per_year = ctx.periods_per_year
    shrunk_band = ctx.shrunk_band
    applied = ctx.applied
    benchmark = ctx.benchmark
    benchmark_close = ctx.benchmark_close
    last_date = ctx.last_date
    settings_key = ctx.settings_key
    r_all = returns_all["simple_return"]
    tab_backtest, tab_model, tab_advice = st.tabs(["Backtest & signals", "Model", "Advice"])
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
            bt_key = ctx.data_key
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
                    "estimated quantity (such as a Kelly fraction) is fitted on the same period it is evaluated on."
                )
            columns["Buy & hold — out-of-sample span"] = bh.out_of_sample.metrics
            columns["Kelly buy & hold — out-of-sample"] = kbh.out_of_sample.metrics
            curves["Buy & hold"] = bh.out_of_sample.equity
            curves["Kelly buy & hold (out-of-sample)"] = kbh.out_of_sample.equity

            equity = pd.DataFrame(curves)
            st.plotly_chart(growth_chart(equity))
            st.caption(
                f"How to read this: growth of $1 from {span_start.date()}, the first out-of-sample decision; every "
                "line starts on the same day so they can be compared directly."
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
                "in excess of the T-bill rate."
            )
            st.plotly_chart(drawdowns_chart(equity))
            st.caption("How to read this: how far each strategy sat below its own previous high.")

            st.markdown("**Signal cards (walk-forward, out-of-sample)**")
            cards = signal_cards(list(SIGNALS), run_wf, BUY_AND_HOLD.name, "Kelly buy & hold")
            pct_cols = ["OOS CAGR", "OOS max drawdown", "Exposure", "Buy & hold max DD"]
            st.dataframe(
                cards.table.style.format(
                    {
                        c: ("{:.1%}" if c in pct_cols else "{:.2f}")
                        for c in cards.table.columns
                        if c != "Beats buy-and-hold OOS"
                    }
                )
            )
            st.caption(
                "Every registered signal, sorted by out-of-sample Sharpe; none are hidden. 'Beats buy-and-hold OOS' "
                "compares Sharpe on the same span; check max drawdown too."
            )
            st.metric("Trials across all signals", cards.trials)
            st.caption(TRIALS_CAPTION)

    with tab_model:
        st.caption(
            f"Probability that {ticker} beats the T-bill rate over the next H trading days, from LightGBM trained "
            "walk-forward (each year predicted by a model trained only on earlier data) and calibrated on "
            "held-out data."
        )
        horizon = st.selectbox("Horizon H (trading days)", [21, 5], key="model_horizon")
        model_prices = close_all.loc[rf_all.index[0] :] if len(rf_all) else close_all
        model_key = ctx.data_key
        report = model_cached(
            ticker, last_date, model_key, int(horizon), benchmark, model_prices, rf_all, benchmark_close
        )
        if isinstance(report, str):
            st.info(f"The model is unavailable: {report}. Choose a longer History.")
        else:
            if report.has_skill:
                st.success(
                    f"Out-of-sample Brier {report.metrics['Brier']:.4f} beats climatology "
                    f"{report.metrics['Brier (climatology)']:.4f}: some demonstrated skill."
                )
            else:
                st.warning(
                    "Model has no demonstrated skill for this ticker: its out-of-sample Brier score "
                    f"({report.metrics['Brier']:.4f}) is not better than always predicting the training base rate "
                    f"({report.metrics['Brier (climatology)']:.4f}). Its probability is treated as 0.5 in the advice."
                )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest P(beat T-bill)", f"{report.latest_probability:.0%}")
            m2.metric("AUC (out-of-sample)", f"{report.metrics['AUC']:.3f}")
            m3.metric(
                "Log loss",
                f"{report.metrics['Log loss']:.3f}",
                f"climatology {report.metrics['Log loss (climatology)']:.3f}",
                delta_color="off",
            )
            m4.metric("Out-of-sample predictions", f"{int(report.metrics['Out-of-sample predictions']):,}")
            st.metric("Trials (hyperparameter combinations)", report.trials)
            st.caption(TRIALS_CAPTION)
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(reliability_chart(report.reliability))
                st.caption(
                    "How to read this: points on the dashed line are well calibrated; bigger points hold more "
                    "predictions. Counts per bin are in the table below."
                )
            with r2:
                st.plotly_chart(contributions_chart(report.latest_contributions))
                st.caption("How to read this: features pushing the latest prediction up (right) or down (left).")
            st.dataframe(report.reliability.set_index("bin"))
    with tab_advice:
        st.warning(BANNER)
        a1, a2, a3 = st.columns(3)
        current_position = a1.slider("Your current position (fraction of portfolio)", 0.0, 1.0, 0.0, 0.05)
        p_buy = a2.slider("p_buy", 0.50, 0.75, 0.55, 0.01)
        p_sell = a3.slider("p_sell", 0.25, 0.50, 0.45, 0.01)
        advice_horizon = int(st.session_state.get("model_horizon", 21))
        advice_prices = close_all.loc[rf_all.index[0] :] if len(rf_all) else close_all
        advice_key = ctx.data_key
        advice_report = model_cached(
            ticker, last_date, advice_key, advice_horizon, benchmark, advice_prices, rf_all, benchmark_close
        )
        if isinstance(advice_report, str):
            st.info(f"Advice needs the probability model, which is unavailable: {advice_report}.")
        else:
            p_model = advice_probability(advice_report)
            regime_labels, _ = regimes_cached(ticker, last_date, settings_key, prior_sd_annual, r_all, rf_all)
            known_regimes = regime_labels.dropna()
            current_regime = str(known_regimes.iloc[-1]) if len(known_regimes) else None
            inputs = AdviceInputs(
                p=p_model,
                kelly_p5=shrunk_band.p5,
                target=applied(raws["Shrunk"]),
                regime=current_regime,
                dd_kelly=raws["Drawdown-constrained"],
                current=current_position,
            )
            advice = decide(inputs, p_buy, p_sell)
            st.subheader(f"{advice.action} → target position {advice.new_position:.0%}")
            if not advice_report.has_skill:
                st.info("The model has no demonstrated out-of-sample skill for this ticker, so p is set to 0.5.")
            st.dataframe(pd.DataFrame({"Fired": advice.clauses}).rename_axis("Rule clause"))
            st.caption(
                f"Inputs: p = {inputs.p:.2f} (H = {advice_horizon} days), shrunk Kelly 5th percentile = "
                f"{inputs.kelly_p5:.2f}, applied fractional Kelly target = {inputs.target:.0%}, regime = "
                f"{inputs.regime or 'unknown'}, drawdown-constrained Kelly = {inputs.dd_kelly:.0%}. SELL / TRIM is "
                "checked first and moves to the lowest cap that fired (target, 0, or drawdown-constrained Kelly)."
            )
            contributions = advice_report.latest_contributions
            top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(5)
            st.markdown("**Top drivers of the model probability**")
            st.dataframe(top.rename("Contribution to log-odds").to_frame())

            st.markdown("**Out-of-sample track record of this exact rule**")
            gate = causal_skill_gate(
                advice_report.probabilities, advice_report.labels, advice_report.climatology, advice_horizon
            )
            rule_settings = (float(k_fraction), float(prior_sd_annual), max_dd, float(p_buy), float(p_sell))
            record = rule_track_record_cached(
                ticker,
                last_date,
                advice_key,
                advice_horizon,
                rule_settings,
                DEFAULT_COST_BPS,
                periods_per_year,
                gate,
                advice_prices,
                rf_all,
            )
            baseline_specs = {
                "Buy & hold": (BUY_AND_HOLD.name, ("", 0.0, 0.0, 0.0)),
                "Kelly buy & hold": ("Kelly buy & hold", (method, float(k_fraction), float(prior_sd_annual), max_dd)),
            }
            baselines = {
                label: walk_forward_cached(
                    ticker,
                    last_date,
                    advice_key,
                    name,
                    settings,
                    DEFAULT_COST_BPS,
                    0.0,
                    "expanding",
                    periods_per_year,
                    advice_prices,
                    rf_all,
                )
                for label, (name, settings) in baseline_specs.items()
            }
            record_table = pd.DataFrame(
                {"Advice rule": record.out_of_sample.metrics}
                | {name: wf.out_of_sample.metrics for name, wf in baselines.items()}
            ).loc[["CAGR", "Sharpe", "Max drawdown", "Exposure", "Annual turnover"]]
            st.dataframe(record_table.style.format("{:.2f}"))
            st.caption(
                "Walk-forward: Kelly inputs are estimated on each training window; p is used only after the model's "
                "own past predictions (with known outcomes) beat climatology, else 0.5. "
                f"Costs {DEFAULT_COST_BPS:g} bps per side."
            )
            st.plotly_chart(
                growth_chart(
                    pd.DataFrame(
                        {"Advice rule": record.out_of_sample.equity}
                        | {name: wf.out_of_sample.equity for name, wf in baselines.items()}
                    )
                )
            )
            st.caption("How to read this: growth of $1 for the rule and both baselines over the same span.")

            st.markdown("**Forward fan**")
            fan = fan_cached(ticker, last_date, advice_key, round(inputs.target, 4), advice_prices, rf_all)
            st.plotly_chart(fan_chart(fan))
            st.caption(
                f"Resampled history, not a forecast: 1,000 one-year paths at the applied fraction {inputs.target:.0%}, "
                "drawn in 20-day blocks from this stock's own past returns."
            )
