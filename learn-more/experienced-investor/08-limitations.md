# 8. Limitations and known approximations

[← Advice rules](07-advice-rules.md) · [Track overview](README.md) · Plain-language version: [Limits and staying safe](../new-investor/08-limits-and-staying-safe.md)

A consolidated list of what the app approximates or ignores, so results can be read with the right discount.

## Data

- **Survivorship bias.** Only currently listed tickers can be analysed, which biases historical comparisons upward.
- **Free data source.** Yahoo Finance via `yfinance` can have gaps or revisions. Restatements of the adjusted series
  are detected and trigger a full refetch ([page 1](01-data-and-conventions.md)); other errors aren't detected.
- **Risk-free rate.** `^IRX` is a discount-basis yield treated as an effective annual rate (understates by about
  0.17 pp at 4%). It is the USD rate for every ticker and base currency, so for other currencies excess returns and
  Kelly are measured against the wrong hurdle.
- **Annualization.** Calendar-based session counts for single-asset metrics, but a fixed 252 inside the shrinkage
  prior `τ`, and 252/52 on the portfolio page.
- **Factor data** is U.S.-only and lags by several weeks.

## Estimation and sizing

- **Stationarity.** Every estimator, and every bootstrap, assumes the lookback window represents the future. Block
  bootstrap (fixed 20-day blocks) preserves only short-range dependence; regime shifts longer than a block aren't
  captured.
- **Growth figure.** The annualized growth number is a second-order approximation; it is inaccurate for large `f` or
  heavy tails.
- **Parametric VaR** assumes normality. It is shown mainly for its gap to historical VaR.
- **Grids.** Log-optimal Kelly uses a 0.005 grid on [0, 10]; drawdown-constrained Kelly uses 5% steps, a fixed 1-year
  horizon and 90% confidence.
- **Prior.** One prior SD for every asset, centred on zero excess return.

## Testing

- **Multiple testing.** Trials are counted and shown, but Sharpe ratios aren't deflated for them.
- **Execution.** Trades happen at the close with flat per-side costs and optional slippage; there is no market impact,
  borrow cost, tax or dividend-timing model. Signals are long-only between 0 and 1.
- **One path.** Walk-forward results describe a single historical path; there is no confidence interval on
  out-of-sample metrics.
- **Regime labels** use expanding percentiles, so early labels rest on less history (at least 252 rolling values).

## Probability model and advice

- **Weak evidence of skill.** Overlapping labels shrink the effective sample size. The skill gate (a cumulative
  Brier-vs-climatology comparison over at least 252 scored predictions, applied identically live and in the track
  record) is a minimum bar, not proof of skill; 252 overlapping 21-day labels carry far fewer independent outcomes.
- **HMM regimes** look ahead and are display-only by design.
- **Fan chart** is resampled history, not a forecast.

## Portfolio

- **Single period.** Mean–variance Kelly with shrunk μ and Ledoit–Wolf Σ; transaction costs are reported but aren't
  part of the optimization.
- **Sector caps** rely on Yahoo's sector field; a failed lookup leaves that ticker uncapped.
- **Failures.** An optimizer failure yields no allocation, by design ([page 4](04-portfolio-construction.md)).

## Scope

No taxes, no account-specific constraints, no personal circumstances. The app is an educational tool and doesn't give
investment advice.

## References

- [Survivorship bias](https://en.wikipedia.org/wiki/Survivorship_bias) and
  [Data dredging](https://en.wikipedia.org/wiki/Data_dredging) (Wikipedia)
- Bailey and López de Prado, "The Deflated Sharpe Ratio", *Journal of Portfolio Management* 40(5), 2014,
  [doi:10.3905/jpm.2014.40.5.094](https://doi.org/10.3905/jpm.2014.40.5.094)

[← Advice rules](07-advice-rules.md) · [Track overview](README.md)
