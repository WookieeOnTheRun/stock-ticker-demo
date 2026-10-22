# 7. The probability model and the advice

[← Testing ideas on the past](06-testing-ideas-on-the-past.md) · [Glossary](glossary.md) · Next: [Limits and staying safe →](08-limits-and-staying-safe.md)

The **Model** and **Advice** tabs on the Backtest & Signals page produce the app's BUY / SELL / HOLD suggestion. This
page explains where it comes from, and why it is deliberately hard to get a BUY.

## The probability model

The model answers one question: **what is the chance this stock beats the T-bill rate over the next 21 trading days**
(about a month; you can also choose 5 days)?

To answer it, the app describes each past day with a handful of measurements ("features"): recent returns over several
periods, how far the price is from its moving averages, recent volatility, the volatility regime, how far it is below
its yearly high, and, if a benchmark is available, how closely it moves with the market and how it has done
compared with it. It then trains a
machine-learning method called **LightGBM**, which builds many small decision trees, to learn how those measurements
related to what happened next.

Three safeguards keep this honest:

1. **Walk-forward training.** Like the backtests on [page 6](06-testing-ideas-on-the-past.md), each year is predicted
   by a model trained only on earlier years.
2. **Calibration.** A raw machine-learning score isn't a true probability. The app adjusts it on held-back data, so
   that "60%" really means "happened about 60% of the time".
3. **A skill test.** The model is compared with a know-nothing forecast: always predicting the long-run share of
   periods in which the stock beat T-bills, the "base rate". The comparison uses a standard accuracy score for
   probabilities (the **Brier score**) on the out-of-sample years.

## "No demonstrated skill" is a common and honest result

Predicting short-term stock moves is extremely hard; professional investors with vast resources mostly fail at it. So
it is normal for the model to **fail the skill test**. When it does, the app says so clearly, and the advice treats the
model's probability as **50%**, a coin flip, instead of trusting a forecast that has no track record.

Beating the know-nothing forecast isn't enough on its own, either. The model also needs at least **252 past predictions
whose outcomes are already known** (about a year of trading days) before the advice uses its probability. Until then,
a good-looking score could easily be luck, so the advice still uses 50%, and the app tells you why. This is exactly
the same test the app uses when it checks the rule's past track record (below), so the live advice never trusts the
model sooner than that check would have.

## The advice rule

The suggestion comes from a short, fixed set of rules that you can read in full. It combines four inputs:

- **p**: the model's probability (or 50% if it hasn't yet proven its skill)
- the **Shrunk Kelly target**: the applied Kelly fraction from [page 3](03-how-much-to-invest-kelly.md), and the
  bottom (5th percentile) of its bootstrap range
- the current **volatility regime** (low / normal / high)
- the **drawdown-constrained Kelly** fraction

You also enter **your current position** (how much of your money is already in the stock).

**SELL / TRIM** is checked first, if *any* of these is true:

- you hold more than the Kelly target plus 5 percentage points (cut back to the target)
- p is at or below **p_sell** (default 45%) (sell everything)
- the market is in a **high-volatility** regime and your position is above the drawdown-constrained amount (cut back to
  that amount)

If several apply, the app moves to the lowest of those levels.

**BUY / ADD** applies only if *all* of these are true:

- even the bottom of the Shrunk Kelly range is above zero (the edge is reasonably clear)
- p is at or above **p_buy** (default 55%)
- the regime is **not** high-volatility

Then the suggested position is the Kelly target. Otherwise the answer is **HOLD**: keep what you have.

Notice what this means with the default settings: if the model has no skill, p is 50%, which is below the 55% needed
for BUY. **So without a model that has proven itself, the app will not say BUY.** It may still say SELL / TRIM, for
example if you hold too much or markets turn stormy. This caution is intentional.

The table under the suggestion lists every rule and whether it fired, so you can always see *why*.

## Has this exact rule worked before?

Below the suggestion, the app backtests **this same rule** walk-forward, with the settings you chose. It lets the model
count only after its own past predictions had proven themselves (at least 252 scored predictions, beating the base
rate). It compares the result with buy-and-hold. If the rule didn't beat simply holding, weigh that heavily.

## The forward fan

Finally, the **fan chart** shows a range of possible one-year outcomes at the suggested position. It is built from
1,000 reshuffled versions of the stock's own past. The app labels it **"resampled history, not a forecast"**: it shows
how wide the range of outcomes *has* been, not what *will* happen.

## Want to go deeper?

- [Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) (the family LightGBM belongs to),
  [Brier score](https://en.wikipedia.org/wiki/Brier_score),
  [Probabilistic classification and calibration](https://en.wikipedia.org/wiki/Probabilistic_classification) and
  [Base rate](https://en.wikipedia.org/wiki/Base_rate) (Wikipedia)
- [Probability calibration (scikit-learn user guide)](https://scikit-learn.org/stable/modules/calibration.html)
- Features, cross-validation and the exact decision rule:
  [The probability model](../experienced-investor/06-probability-model.md) and
  [Advice rules](../experienced-investor/07-advice-rules.md) (experienced track)

[← Testing ideas on the past](06-testing-ideas-on-the-past.md) · Next: [Limits and staying safe →](08-limits-and-staying-safe.md)
