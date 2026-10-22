# 7. Advice rules

[← The probability model](06-probability-model.md) · Next: [Limitations →](08-limitations.md) · Plain-language version: [The model and the advice](../new-investor/07-the-model-and-the-advice.md)

The advice is a short deterministic rule (`stockapp/advice.py`) rather than another model. Every clause is reported, so
each suggestion is fully explainable, and the rule itself is backtested walk-forward next to it.

## Inputs on the Advice tab

| Input | Source |
|---|---|
| `p` | Latest calibrated probability at the Model tab's horizon if `skill_established` (below), else 0.5 |
| `target` | Applied Shrunk Kelly on the lookback window: `clip(k·f*_shrunk, 0, 1)`. Always the Shrunk estimate, whatever method the sidebar shows. |
| `kelly_p5` | 5th percentile of the shrunk-`f*` block bootstrap on the lookback ([page 2](02-kelly-sizing.md)) |
| `regime` | Latest causal volatility regime on the full history ([page 3](03-risk-and-distribution.md)) |
| `dd_kelly` | Drawdown-constrained fraction on the lookback (not multiplied by `k`) |
| `current` | Your current position, entered on the tab |

## Decision rule

```
SELL / TRIM if  current > target + 0.05                          → cap: target
             or p ≤ p_sell                                        → cap: 0
             or regime == "high" and dd_kelly < current           → cap: dd_kelly
             new position = min(current, every cap that fired)
else BUY / ADD if kelly_p5 > 0 and p ≥ p_buy and regime != "high" → new position: target
else HOLD                                                         → position unchanged
```

Defaults are `p_buy = 0.55` (slider 0.50–0.75) and `p_sell = 0.45` (slider 0.25–0.50). SELL is checked first, so an
over-target position is trimmed even when the buy clauses hold.

**Design consequences**

- With defaults and no demonstrated skill (p = 0.5), **BUY can't fire**. The rule then only trims toward the Kelly
  target or toward `dd_kelly` in high volatility. Entry requires both a statistically non-negative edge
  (`kelly_p5 > 0`) and a model with demonstrated skill that agrees.
- `kelly_p5 > 0` is a one-sided ~95% requirement on the sign of the edge under the block bootstrap. It is much
  stricter than `f* > 0`.
- The high-volatility clause reduces exposure when it matters most, using the user's own drawdown tolerance.

## Walk-forward track record

The same rule is run as a walk-forward strategy over the full history (expanding folds, 5 bps per side), next to
buy-and-hold and Kelly buy-and-hold. To keep it free of look-ahead:

- **Kelly inputs** (`target`, `kelly_p5`, `dd_kelly`) are estimated on each fold's whole training window and frozen
  for its test year. The regime is read daily from causal labels.
- **Causal skill gate** (`causal_skill_gate`): on each date the model's p is used only if, among predictions whose
  outcomes were already known by that date (label end ≤ date), there are at least 252, and their cumulative Brier
  score beats climatology's. Otherwise p = 0.5.

**The live tab uses the same gate.** `skill_established(report)` evaluates this gate on the latest decision date,
calling the same code, so the live advice acts on p exactly when the track record would have on that date. On the
latest date every known outcome counts, so the gate reduces to `has_skill` ([page 6](06-probability-model.md)) plus
the 252-prediction minimum. When a model beats climatology on fewer predictions, the Model and Advice tabs say that
p is 0.5 for that reason. The [scanner](#watchlist-scanner-python--m-stockappscan) applies the same gate, and its
`model_has_skill` column reports it.

## Forward fan

A Monte Carlo of one-year equity at a constant fraction `target`: 1,000 paths of 252 days, block-bootstrapped (20-day
blocks, seed 42) from the observed `(r, rf)` pairs over the full history, with equity `cumprod(1 + rf + f(r − rf))`.
The chart shows the 5/25/50/75/95th percentile paths. It is resampled history, not a forecast: it assumes the past
return distribution and its short-range dependence persist.

## Watchlist scanner (`python -m stockapp.scan`)

The scanner runs the same `decide()` for every ticker in a watchlist CSV, through the cached data layer. Prices are in
USD, with 10 years of history, a 252-day Kelly lookback and a 21-day horizon by default, plus fixed `k = 0.5`, prior
SD 0.10 and max drawdown 25%. `current` comes from the CSV's `weight` column. It writes an escaped HTML report plus
JSON and, with `--email`, sends only the tickers whose action changed (SMTP over STARTTLS, credentials from the
environment). See the [project README](../../README.md#watchlist-scanner) for usage and scheduling.

## References

- [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method) and
  [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) (Wikipedia)
- [Brier score](https://en.wikipedia.org/wiki/Brier_score) and [Base rate](https://en.wikipedia.org/wiki/Base_rate)
  (Wikipedia)

[← The probability model](06-probability-model.md) · Next: [Limitations →](08-limitations.md)
