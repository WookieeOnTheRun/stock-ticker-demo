# 6. Testing ideas on the past

[← Building a portfolio](05-building-a-portfolio.md) · [Glossary](glossary.md) · Next: [The model and the advice →](07-the-model-and-the-advice.md)

The **Backtest & Signals** page answers: *if I had followed this rule in the past, how would it have gone?* That
replay is called a **backtest**. Backtests are useful, and also one of the easiest ways to fool yourself, so this page
explains both.

## How a backtest works here

A **strategy** is a rule that decides, at the end of each trading day, what share of your money should be in the
stock (from 0% to 100%); the rest sits in T-bills. The app replays that rule over the history, with three fairness
rules built in:

1. **No peeking.** A decision made at today's close only earns *tomorrow's* return. Otherwise a rule could "buy"
   using a price it couldn't have known in time.
2. **Trading costs.** Every change in position is charged a cost (default 0.05% per side, adjustable, plus optional
   "slippage"). Rules that trade a lot pay for it.
3. **Cash earns interest.** Money out of the stock earns the T-bill rate, as it would in real life.

## The trap: tuning a rule until it looks good

Most strategies have settings, such as "use a 50-day average" vs "a 200-day average". If you try many settings and
keep whichever did best on past data, you'll almost always find something that looks great. It often looks great by
**luck**, fitted to the quirks of that particular history. This is **overfitting**, and it is the main reason
impressive backtests disappoint in real life.

## The defence: walk-forward testing

The app's default is **walk-forward** testing, which mimics what you could really have done:

1. Take the first **3 years** of data. Try each of the strategy's settings on those 3 years only and pick the best.
2. **Freeze** that choice and use it for the **next year**, data it has never seen.
3. Move forward a year and repeat (the training period either keeps growing, "expanding", or stays the most recent 3
   years, "rolling").

Stitching those "never seen before" years together gives an **out-of-sample** record: the fairest estimate of how the
rule might do going forward. The page shows it next to the **in-sample** result (how the chosen settings did on the
data used to choose them), so you can see how much the past flattered the rule. There is also a **fixed parameters**
mode, clearly labelled as in-sample, for exploring.

## Counting how many things were tried

The app shows **Trials**: how many setting combinations were tested. The more things you try, the more likely the best
one is a fluke, so a strategy that only "wins" after many trials deserves extra suspicion.

## The comparison that matters: buy and hold

Every strategy is compared with two simple baselines over the same period:

- **Buy & hold:** fully invested every day.
- **Kelly buy & hold:** a constant position at the Kelly-sized amount, worked out only from earlier data.

If a clever rule can't beat simply holding, the clever rule isn't worth its complexity and trading costs.

## The signal library

The page includes several classic, well-known trading rules ("signals"):

| Signal | Idea |
|---|---|
| **SMA cross** | Be invested while the short-term average price is above the long-term average (an uptrend). |
| **Donchian breakout** | Buy when the price breaks above its recent high; sell when it falls below its recent low. |
| **Momentum 12-1** | Be invested if the stock rose over the past year (ignoring the most recent month). |
| **RSI mean reversion** | Buy when the stock has fallen a lot recently ("oversold"); sell once it has bounced ("overbought"). |
| **Bollinger z** | Buy when the price is unusually far below its recent average; sell when it gets back to the average. |
| **Vol target** | Stay invested but hold less when the stock is very jumpy, to keep risk roughly steady. |
| **Regime filter** | Step out of the stock during high-volatility periods (see [page 4](04-reading-the-risk-tabs.md)). |

The **signal cards** table shows *every* signal's out-of-sample result, sorted best to worst. Losers are never hidden,
because hiding them is another way backtests mislead. "Beats buy-and-hold OOS" compares the Sharpe ratio; check the
drawdown column too, since a rule with a lower Sharpe might still have avoided big falls.

## Want to go deeper?

- [Backtesting](https://en.wikipedia.org/wiki/Backtesting),
  [Overfitting](https://en.wikipedia.org/wiki/Overfitting),
  [Walk-forward optimization](https://en.wikipedia.org/wiki/Walk_forward_optimization) and
  [Data dredging](https://en.wikipedia.org/wiki/Data_dredging), the "try lots of things" problem (Wikipedia)
- The signals: [Moving average](https://en.wikipedia.org/wiki/Moving_average),
  [Donchian channel](https://en.wikipedia.org/wiki/Donchian_channel),
  [Momentum](https://en.wikipedia.org/wiki/Momentum_(finance)),
  [Relative strength index](https://en.wikipedia.org/wiki/Relative_strength_index),
  [Bollinger Bands](https://en.wikipedia.org/wiki/Bollinger_Bands) and
  [Mean reversion](https://en.wikipedia.org/wiki/Mean_reversion_(finance)) (Wikipedia)
- Exact rules, parameter grids and the backtest equation: [Backtesting and signals](../experienced-investor/05-backtesting-and-signals.md)
  (experienced track)

[← Building a portfolio](05-building-a-portfolio.md) · Next: [The model and the advice →](07-the-model-and-the-advice.md)
