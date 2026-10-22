# 5. Backtesting and signals

[← Portfolio construction](04-portfolio-construction.md) · Next: [The probability model →](06-probability-model.md) · Plain-language version: [Testing ideas on the past](../new-investor/06-testing-ideas-on-the-past.md)

## Engine (`stockapp/backtest/engine.py`)

A strategy maps prices to a target weight `w_t ∈ [0, 1]`, decided at the close of `t` using data through `t`. Daily
strategy return:

```
R_t = w_{t−1}·r_t + (1 − w_{t−1})·rf_t − (cost_bps + slippage_bps)/10⁴ · |w_t − w_{t−1}|
```

The one-day shift is what prevents a close-`t` decision from earning day `t`'s return. Missing targets mean cash.
Costs default to 5 bps per side with 0 slippage, charged on every change in weight. The trade decided at the first
close is charged on the second day.

**Metrics:** CAGR, annualized volatility, Sharpe and Sortino (on excess returns), max drawdown and its duration, hit
rate (invested days beating rf), profit factor, average holding period, exposure and annual turnover (a full
round-trip counts 2.0).

## Walk-forward evaluation (`stockapp/backtest/walkforward.py`)

- **Folds:** test year *k* starts `3 + k` years after the first date. Training is everything before it (expanding,
  the default) or the preceding 3 years (rolling). The last test segment may be shorter.
- **Selection:** each grid point is backtested on the training window only, and the best **training Sharpe** wins
  (first wins ties). Strategies with an `estimate` step (e.g. a Kelly fraction) fit it on the training window only
  and merge it into every grid point.
- **Stitching:** fold *k*'s parameters set the target at every close from its last training date up to (not
  including) the next fold's last training date. Positions are computed on prices through each fold's test end, so
  indicators warm up on real history. The stitched positions are backtested **once**, so costs at fold boundaries are
  charged.
- **In-sample comparison:** the final fold's parameters run over the training window that chose them.
- **Trials:** the grid size is reported for every strategy and summed on the signal cards. A best-of-N result needs a
  higher bar as N grows; see the deflated Sharpe ratio reference below.
- **Fixed-parameter mode** runs one grid point over the whole span. It is labelled in-sample, since any estimated
  quantity is fitted on the period it is evaluated on.

Causality is enforced by `tests/test_no_lookahead.py`. For every registered signal and grid point, every model feature
and the walk-forward harness, it recomputes the output on the series truncated at 20 random dates. Any change to
values at or before the cut fails the test. A deliberately leaky function is included to prove the guard catches leaks.

## Signals (`stockapp/signals/library.py`)

| Signal | Rule (target weight) | Grid |
|---|---|---|
| Buy & hold | 1 | — |
| Kelly buy & hold | Constant `clip(k·f*, 0, 1)`, `f*` from the sidebar method on the training window | — |
| SMA cross | 1 while SMA(fast) > SMA(slow) | fast ∈ {20, 50} × slow ∈ {100, 200} |
| Donchian breakout | Enter on close > prior n-day high; exit on close < prior n/2-day low (channels lagged 1 day) | n ∈ {55, 100} |
| Momentum 12-1 | 1 if `P_{t−21}/P_{t−252} − 1 > 0` | — |
| RSI mean reversion | Wilder RSI(14) (EWM, α = 1/14): enter below `lo`, exit above `hi` | lo ∈ {25, 30} × hi ∈ {70, 75} |
| Bollinger z | `z = (P − SMA₂₀)/SD₂₀`: enter at z < −z₀, exit at z > 0 | z₀ ∈ {1.5, 2.0} |
| Vol target | `min(1, target / EWMA vol)` (λ = 0.94) | target ∈ {10%, 15%} |
| Regime filter | 0 in the causal high-volatility regime, else 1 (invested before labels exist) | — |

Entry/exit signals hold a position from an entry close until the next exit close; exits win ties. The HMM regimes are
never used ([page 3](03-risk-and-distribution.md)). No earnings-drift signal is registered, because the surprise data
it needs isn't available.

**Signal cards** list every signal's out-of-sample Sharpe, CAGR, max drawdown, exposure and turnover. Each sits beside
buy-and-hold and Kelly buy-and-hold on the same span, sorted by out-of-sample Sharpe, with none dropped. "Beats
buy-and-hold OOS" means a higher out-of-sample Sharpe; drawdown is shown alongside, so a lower-risk loser on Sharpe
stays visible.

## References

- D. H. Bailey and M. López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting
  and Non-Normality", *Journal of Portfolio Management* 40(5), 2014,
  [doi:10.3905/jpm.2014.40.5.094](https://doi.org/10.3905/jpm.2014.40.5.094)
- Wikipedia: [Backtesting](https://en.wikipedia.org/wiki/Backtesting),
  [Walk-forward optimization](https://en.wikipedia.org/wiki/Walk_forward_optimization),
  [Data dredging](https://en.wikipedia.org/wiki/Data_dredging),
  [Transaction cost](https://en.wikipedia.org/wiki/Transaction_cost),
  [Momentum](https://en.wikipedia.org/wiki/Momentum_(finance)),
  [Relative strength index](https://en.wikipedia.org/wiki/Relative_strength_index),
  [Bollinger Bands](https://en.wikipedia.org/wiki/Bollinger_Bands),
  [Donchian channel](https://en.wikipedia.org/wiki/Donchian_channel)

[← Portfolio construction](04-portfolio-construction.md) · Next: [The probability model →](06-probability-model.md)
