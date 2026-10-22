# 4. Reading the risk and analysis tabs

[← How much to invest](03-how-much-to-invest-kelly.md) · [Glossary](glossary.md) · Next: [Building a portfolio →](05-building-a-portfolio.md)

The Research page has a row of tabs. The Kelly number from [page 3](03-how-much-to-invest-kelly.md) is only as good as
the assumptions behind it; these tabs help you check them. This page walks through each tab in plain language.

## Risk tab: how bad can it get?

| Measure | What it tells you |
|---|---|
| **1-day VaR 95% (historical)** | "Value at risk." On a typical bad day, 1 day in 20, you lost at least this much. |
| **1-day CVaR 95% (historical)** | The *average* loss on those worst 1-in-20 days. Always at least as big as VaR; it tells you how bad the bad days are. |
| **1-day VaR 95% (normal)** | The same idea as VaR, but calculated as if returns followed a neat bell curve. If it is much smaller than the historical VaR, the bell curve is underestimating the risk. |
| **Max drawdown** | The deepest fall from a peak in the lookback window, with the dates it started, bottomed out and recovered. |
| **Sortino** | Like the Sharpe ratio, but only counts *downward* swings as risk. |
| **Calmar** | Yearly growth (CAGR) divided by the maximum drawdown: how much growth per unit of worst-case pain. |
| **Omega** | Total gains on up days divided by total losses on down days. Above 1 means gains outweighed losses. |
| **Beta** | How much the stock tends to move when the benchmark (by default the S&P 500 fund SPY) moves. Beta 1.5 means it has typically moved about 1.5% for each 1% market move. |

The tab also shows **volatility over time**, two ways: a simple 21-day average, and a version that reacts faster to
recent shocks (EWMA). It also tries a **GARCH** forecast of volatility for the next 21 days. GARCH is a widely used
model that captures the fact that stormy days tend to follow stormy days. If GARCH can't be fitted, the app says so
and uses the faster-reacting estimate instead.

## Distribution tab: are the returns "well-behaved"?

The simplest Kelly formula (Gaussian) assumes daily returns follow a **bell curve** (a "normal distribution"). Real
stock returns usually don't, in two ways that matter:

- **Fat tails:** extreme days, crashes and spikes, happen far more often than a bell curve says.
- **Volatility clustering:** calm periods and turbulent periods come in spells, not randomly mixed.

The tab answers **"Is Gaussian Kelly safe for this stock?"** with a plain sentence. It runs standard statistical tests
for both problems. If either shows up, it suggests preferring the Shrunk or Log-optimal estimate, because Gaussian
Kelly tends to overstate the safe amount when tails are fat. It also compares the Log-optimal and Gaussian answers; a
big gap is the price of those fat tails.

The **Q-Q charts** compare the real returns with a bell curve: if the dots bend away from the dashed line at both
ends, the tails are fatter than the bell curve. The **Hurst exponent** hints at whether moves tend to continue
(above 0.5), reverse (below 0.5) or neither (about 0.5); the app labels it "indicative" because it is unreliable on
short histories.

## Regimes tab: calm markets vs stormy markets

A **regime** is a period with a particular "mood". The app labels each day **low**, **normal** or **high** volatility.
It compares that day's recent bumpiness (the last 21 days) with all the earlier days: bottom third is "low", top third
is "high". Importantly, each label only uses information available on that day, so labels never "know the future".
The table then shows the Kelly fraction in each regime. Many stocks look much less attractive in high-volatility
spells.

There is also an optional **HMM** view, a statistical model that finds regimes using the whole history at once. Because
it uses later data to label earlier days, the app shows it as a *historical view only* and never uses it for any
decision.

## Factors tab: what is really driving this stock?

Researchers have found a handful of broad forces, called **factors**, that explain a lot of how stocks move: the
overall market, small vs large companies, cheap vs expensive ("value"), profitable vs unprofitable companies,
companies that invest conservatively vs aggressively, and recent winners vs losers ("momentum"). The app downloads
these from a respected academic data library and measures how strongly the stock is tied to each. It summarizes the
result in one sentence, such as "most of this stock's movement is market beta". The factors are U.S.-based, so the
app warns that the fit means less for non-U.S. stocks.

## Relative tab: compared with what?

Shows how the stock did against your **benchmark** and, where available, the fund for its **sector** (for example the
technology sector fund XLK for a tech company), over the last 1, 3, 6 and 12 months. It also shows a **seasonality**
heatmap of monthly returns. The app itself notes that seasonal patterns in single stocks are mostly noise.

## Want to go deeper?

- [Value at risk](https://en.wikipedia.org/wiki/Value_at_risk),
  [Expected shortfall (CVaR)](https://en.wikipedia.org/wiki/Expected_shortfall),
  [Sortino ratio](https://en.wikipedia.org/wiki/Sortino_ratio),
  [Calmar ratio](https://en.wikipedia.org/wiki/Calmar_ratio),
  [Omega ratio](https://en.wikipedia.org/wiki/Omega_ratio) and [Beta](https://en.wikipedia.org/wiki/Beta_(finance))
  (Wikipedia)
- [Fat-tailed distributions](https://en.wikipedia.org/wiki/Fat-tailed_distribution),
  [Volatility clustering](https://en.wikipedia.org/wiki/Volatility_clustering) and
  [GARCH / ARCH models](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity) (Wikipedia)
- [Fama–French factor model](https://en.wikipedia.org/wiki/Fama%E2%80%93French_three-factor_model) (Wikipedia) and the
  [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), the
  source of the factor data
- The tests, thresholds and formulas: [Risk and distribution diagnostics](../experienced-investor/03-risk-and-distribution.md)
  (experienced track)

[← How much to invest](03-how-much-to-invest-kelly.md) · Next: [Building a portfolio →](05-building-a-portfolio.md)
