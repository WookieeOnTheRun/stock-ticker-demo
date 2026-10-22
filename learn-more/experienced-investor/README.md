# Experienced investor track

How the Fractional Kelly Stock Analyzer estimates, sizes, tests and decides: the formulas, the parameter values, and
the reasoning behind each design choice. Every value quoted here is the value in the code; module paths are given so
you can check them. For plain-language versions, see the [new investor track](../new-investor/README.md).

> **Educational tool, not investment advice.** Past performance does not predict future results.

## Design principles

The same few principles recur on every page:

1. **Estimation error dominates.** Expected returns are the noisiest input, so means are shrunk toward zero, Kelly is
   always fractional, and point estimates come with bootstrap bands.
2. **No look-ahead.** Every feature, label used for a decision, signal and regime is causal at the close it is used.
   Decisions at close *t* earn returns from *t + 1*. A test (`tests/test_no_lookahead.py`) truncates series at random
   dates and fails if any earlier output changes.
3. **Out-of-sample by default.** Strategy parameters and the probability model are chosen walk-forward; trial counts
   are reported.
4. **Long-only, unlevered outputs.** Every recommended position is clipped to [0, 1] (portfolio: `w ≥ 0`,
   `Σw ≤ gross_max ≤ 1`).
5. **Fail visibly.** A model without skill is ignored (p = 0.5); an optimizer failure yields no allocation, never a
   substitute.

## Pages

| # | Page | Covers |
|---|---|---|
| 1 | [Data and conventions](01-data-and-conventions.md) | Prices, caching, risk-free rate, FX, calendars and annualization |
| 2 | [Kelly sizing](02-kelly-sizing.md) | Gaussian, shrunk, log-optimal and drawdown-constrained f\*, block bootstrap bands |
| 3 | [Risk and distribution diagnostics](03-risk-and-distribution.md) | VaR/CVaR, EWMA, GARCH, normality and dependence tests, regimes, factor regression |
| 4 | [Portfolio construction](04-portfolio-construction.md) | Shrunk μ, Ledoit–Wolf Σ, constrained Kelly QP, HRP, bands, frontier, failure handling |
| 5 | [Backtesting and signals](05-backtesting-and-signals.md) | Engine, costs, walk-forward selection, signal definitions and grids |
| 6 | [The probability model](06-probability-model.md) | Features, labels, purged CV, calibration, skill test |
| 7 | [Advice rules](07-advice-rules.md) | The BUY / SELL / HOLD rule, its walk-forward record, fan chart, scanner |
| 8 | [Limitations and known approximations](08-limitations.md) | What is approximated, and what to keep in mind |

Next: [Data and conventions →](01-data-and-conventions.md)

[Back to Learn more](../README.md)
