# 3. How much to invest: the Kelly idea

[← Returns and risk](02-returns-and-risk.md) · [Glossary](glossary.md) · Next: [Reading the risk tabs →](04-reading-the-risk-tabs.md)

This is the heart of the app. The **Kelly criterion** is a rule, published by the Bell Labs scientist John Kelly in
1956, for deciding what share of your money to put into a bet with good odds so that your money grows as fast as
possible over the long run.

## The coin-flip example, worked through

Back to the coin from [page 1](01-the-big-idea.md): heads 60% of the time, and you win or lose exactly what you bet.

Kelly's answer for an even-money bet is simple: **bet the difference between your chance of winning and your chance
of losing.** Here that is 60% − 40% = **20% of your money on each flip.**

- Bet **more** than 20% and, surprisingly, your money grows *more slowly* over many flips, because the losses hurt
  more than the wins help (remember from [page 2](02-returns-and-risk.md) that a 50% loss needs a 100% gain to undo).
  Bet far more and you will very likely go broke, even with the odds on your side.
- Bet **less** than 20% and you grow more slowly too, but more smoothly.

## From coins to stocks

A stock isn't a coin with known odds, but the same logic applies. The app looks at the stock's past **excess returns**
(returns above the T-bill rate) and asks: *what share of my money in this stock would have grown my wealth fastest?*
Roughly, the answer gets **bigger when the stock's average excess return is higher** and **smaller when it is more
volatile**. That answer is called the **Kelly fraction**, written f\*.

A Kelly fraction of 0.40 means "put 40% of your money in this stock and keep 60% in T-bills". A fraction above 1 would
mean borrowing to invest more than you have; a negative fraction would mean betting the price falls. **The app never
suggests either**: it always limits the result to between 0% and 100%.

## Why the app uses only a *fraction* of Kelly

Full Kelly assumes you know the true odds exactly. With stocks you never do: the past average return is a very rough
guess of the future one. If you overestimate your edge, full Kelly makes you bet too much, and over-betting is the
dangerous side.

So the app multiplies the Kelly answer by the **Kelly fraction multiplier** in the sidebar (default **0.5**, "half
Kelly"). A useful rule of thumb from the theory: half Kelly gives you roughly **three-quarters of the growth with half
the bumpiness**. Many professional users of Kelly go lower still.

The number the app shows as **Applied Fractional Kelly** is: Kelly fraction × your multiplier, then limited to between
0% and 100%.

## Four ways to estimate the Kelly fraction

Because the past is a noisy guide, the app offers four estimates (choose one in the sidebar; the Research page shows
all four side by side):

| Method | Plain-language idea | When it helps |
|---|---|---|
| **Gaussian** | The textbook formula: average excess return ÷ variance (volatility squared). | A simple baseline. It takes the past average at face value, so it can swing wildly. |
| **Shrunk** (default) | Same formula, but the average return is first pulled toward zero, more strongly when the data is noisy. | Protects you from being fooled by a lucky streak. This is why it is the default. |
| **Log-optimal** | Tries every fraction from 0 to 10 against the actual past days and picks the one that would have grown money fastest. | Doesn't assume returns follow a neat bell curve, so rare crash days count fully. |
| **Drawdown-constrained** | The largest fraction (in steps of 5%) for which, in 1,000 simulated years built from the stock's own history, the worst fall stays within your **Max tolerable drawdown** setting in at least 90% of them. | Puts your comfort with losses first, instead of maximum growth. |

### What "pulled toward zero" means (the Shrunk method)

Suppose a stock rose a lot last year. Was that skill, or luck? With only a year of data you can't really tell. The
Shrunk method starts from a sceptical assumption, that the stock's true extra return is probably near zero, and only
moves away from that as the evidence gets stronger. The **Prior SD** setting controls how sceptical it is: smaller
values mean "I need more convincing". This idea is called **shrinkage**.

## How sure is the estimate? The bootstrap range

The app also shows a **5–95% range** for the Gaussian and Shrunk estimates. It builds that range by making 1,000
"alternative histories": it cuts the real history into 20-day chunks and reshuffles them, recomputing the Kelly
fraction each time. This technique is called the **bootstrap**. Using chunks rather than single days keeps calm and
stormy periods together, the way they really happen.

- A **narrow** range means the estimate is fairly stable.
- A range that **includes zero** means the data can't even tell whether the stock beat T-bills. The app then warns you
  that Kelly sizing is unreliable for this stock. Take that warning seriously.

## Kelly across different windows

The chart **Applied Kelly Across Lookback Windows** repeats the calculation on the last 60, 120, 252 and 504 trading
days. If the bars differ a lot, the answer depends heavily on which period you look at, which is another sign to be
cautious.

## Want to go deeper?

- [Kelly criterion (Wikipedia)](https://en.wikipedia.org/wiki/Kelly_criterion): the coin example, fractional Kelly and
  the criticisms
- J. L. Kelly's original 1956 paper,
  [A New Interpretation of Information Rate (PDF, Princeton mirror)](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf)
- [Shrinkage](https://en.wikipedia.org/wiki/Shrinkage_(statistics)) and
  [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) (Wikipedia)
- All four formulas and their settings: [Kelly sizing](../experienced-investor/02-kelly-sizing.md) (experienced track)

[← Returns and risk](02-returns-and-risk.md) · Next: [Reading the risk tabs →](04-reading-the-risk-tabs.md)
