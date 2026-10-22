# 8. Limits and staying safe

[← The model and the advice](07-the-model-and-the-advice.md) · [Glossary](glossary.md) · [Track overview](README.md)

Every number in this app comes from the past and from simplifying assumptions. Knowing the limits is part of using it
well.

## What the app cannot know

- **The future.** History can describe how a stock *has* behaved; companies, markets and economies change.
- **You.** It doesn't know your goals, timeline, other savings, debts, income stability or tax situation. A sensible
  position for one person can be reckless for another.
- **Costs that depend on you.** Taxes aren't estimated at all, and trading costs are a simple flat assumption.

## Known blind spots in the data

- **Survivorship bias.** The app can only analyse companies that still trade today. Companies that went bankrupt or
  were delisted are missing, so history looks rosier than it really was.
- **Free data.** Prices come from a free source (Yahoo Finance via the `yfinance` library). It is usually good, but it
  can have gaps or errors, and the app warns you when the latest data looks out of date.
- **One risk-free rate.** The T-bill rate is a U.S. dollar rate. For stocks in other currencies, a local rate would be
  more accurate; the app notes this limitation.
- **Short histories.** With little data, every estimate is shaky. Watch the ranges and warnings.

## Known blind spots in the methods

- The Kelly fraction is very sensitive to the average return, which is the hardest thing to estimate. That is why the
  app shrinks it, scales it down and shows ranges.
- Backtests, even fair walk-forward ones, cover one particular history. A rule that worked in the last ten years may
  not work in the next ten.
- The probability model usually shows **no skill**, and that is the honest result for most stocks.

## Using the app sensibly

1. **Treat outputs as questions, not answers.** "Why is the range so wide?" is often more useful than the number.
2. **Respect the warnings.** "Can't distinguish from zero", "no demonstrated skill" and "may be stale" all mean the
   same thing: be careful.
3. **Prefer smaller.** If unsure, use a lower Kelly multiplier. Betting too little costs a little growth; betting too
   much can cost a lot.
4. **Diversify.** One stock is one company's fate. See [page 5](05-building-a-portfolio.md).
5. **Get personal advice for real decisions.** A licensed professional can consider what this app cannot.

## Want to go deeper?

- [Survivorship bias (Wikipedia)](https://en.wikipedia.org/wiki/Survivorship_bias)
- [Investing basics and glossary (investor.gov)](https://www.investor.gov/introduction-investing/investing-basics/glossary)
- A full list of approximations and caveats: [Limitations and known approximations](../experienced-investor/08-limitations.md)
  (experienced track)

[← The model and the advice](07-the-model-and-the-advice.md) · [Glossary](glossary.md) · [Track overview](README.md)
