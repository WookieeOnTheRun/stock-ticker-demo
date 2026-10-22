# 5. Building a portfolio

[← Reading the risk tabs](04-reading-the-risk-tabs.md) · [Glossary](glossary.md) · Next: [Testing ideas on the past →](06-testing-ideas-on-the-past.md)

The Portfolio page applies the same thinking to several stocks at once (between 2 and 50). You can type tickers, or
upload a small spreadsheet (CSV) that can also list what you already hold.

## Why combine stocks at all? Diversification

If you hold several stocks that don't all move together, a bad day for one is often an ordinary day for another, so
your total swings less than any single stock. This is **diversification**. How much it helps depends on
**correlation**: how closely two stocks move together. Two oil companies are highly correlated, so holding both adds
little protection; an oil company and a software company usually much less so.

The **correlation heatmap** shows this for your stocks; the **cluster tree** groups the ones that move alike.

## The four ways the app splits your money

| Method | Idea in plain words |
|---|---|
| **Equal weight** | The same amount in every stock. Simple and surprisingly hard to beat. |
| **HRP** (Hierarchical Risk Parity) | Groups similar stocks together, then shares the money so each group contributes a similar amount of risk. It doesn't use any guess of future returns, which is a strength, since those guesses are so unreliable. |
| **Fractional Kelly, unconstrained (reference)** | The Kelly idea extended to many stocks with no limits at all. It can suggest borrowing or betting against stocks, so it is shown **for reference only**, never as a suggestion. |
| **Constrained Kelly** | The app's suggestion: the Kelly answer, scaled by your multiplier, but obeying sensible limits you set. |

### The limits on Constrained Kelly

- **No borrowing, no betting against stocks.** Every weight is 0% or more, and the total is at most 100%.
- **Max weight per asset** (default 25%) so no single stock dominates.
- **Max total invested** (default 100%). Whatever isn't invested is held as cash earning the T-bill rate.
- Optionally, a **cap per sector** (default 40% when switched on), so you don't end up with, say, all technology.

Before the calculation, the app also **tames its inputs**. It pulls each stock's expected return toward zero, as in
the Shrunk method of [page 3](03-how-much-to-invest-kelly.md). It also steadies its estimate of how the stocks move
together, because a small amount of data can make that estimate misleading.

## How sure is the answer?

The **Constrained 5% / 95%** columns come from re-running the whole calculation on up to 200 reshuffled versions of the
history (the bootstrap again). Wide ranges mean the suggested weights are fragile.

If the calculation ever fails for numerical reasons, the app **doesn't make up a substitute**. It shows "—" instead of
Constrained Kelly weights, proposes no trades, and explains why. The other methods are unaffected.

## The efficient frontier

The **frontier chart** shows the best mix for each level of risk: each point is the best portfolio for one Kelly
multiplier. Points up and to the left are better: more expected return for less volatility. Your methods are marked on
it so you can compare them. The expected returns are the cautious, shrunk estimates, not forecasts.

## A fair test: first half vs second half

The **growth chart** at the bottom works out each method's weights using only the *first half* of the history, then
shows how those fixed weights did in the *second half*. This is a simple "would it have worked on data it hadn't
seen?" check. (The Backtest page does a more thorough version; see [page 6](06-testing-ideas-on-the-past.md).)

## From suggestion to trades

If you entered your current holdings, the **Current vs target** table shows what you'd buy or sell to reach the
Constrained Kelly mix, and the **turnover** (how much of your portfolio would change hands). If you enter a cost per
trade, it estimates the cost too. The app reminds you that **taxes aren't included**, since they depend on your own
situation.

## Mixing stock exchanges

If your stocks trade on different exchanges (say New York and Hong Kong), their days end hours apart, so their daily
moves don't line up. That makes them look less related than they are. The app therefore suggests **weekly** returns in
that case.

## Want to go deeper?

- [Beginners' guide to asset allocation, diversification and rebalancing (investor.gov)](https://www.investor.gov/additional-resources/general-resources/publications-research/info-sheets/beginners-guide-asset)
- [Diversification](https://en.wikipedia.org/wiki/Diversification_(finance)),
  [Correlation](https://en.wikipedia.org/wiki/Correlation),
  [Modern portfolio theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory),
  [Efficient frontier](https://en.wikipedia.org/wiki/Efficient_frontier) and
  [Hierarchical Risk Parity](https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity) (Wikipedia)
- The optimization problem and estimators: [Portfolio construction](../experienced-investor/04-portfolio-construction.md)
  (experienced track)

[← Reading the risk tabs](04-reading-the-risk-tabs.md) · Next: [Testing ideas on the past →](06-testing-ideas-on-the-past.md)
