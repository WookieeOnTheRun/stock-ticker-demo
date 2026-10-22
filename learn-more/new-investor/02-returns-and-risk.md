# 2. Returns and risk

[← The big idea](01-the-big-idea.md) · [Glossary](glossary.md) · Next: [How much to invest →](03-how-much-to-invest-kelly.md)

Everything in the app is built from a few basic measurements. This page explains each one.

## Price history, adjusted fairly

The app uses each stock's **daily closing price**: the last price of each trading day. It uses the **adjusted** close,
which corrects for two events that would otherwise distort the history:

- A **stock split**, where a company swaps, say, each share for four cheaper ones. The price drops to a quarter
  overnight, but nobody lost money. The adjustment rescales older prices so the split doesn't look like a crash.
- A **dividend**, a cash payment the company makes to shareholders. The adjustment counts it as part of your return,
  as if you had reinvested it.

On the Research page's price chart, dashed lines mark splits and small dots mark dividends.

## Return: how much did it go up or down?

A **return** is the percentage change in price. If a stock closes at $100 one day and $102 the next, that day's return
is +2%. The app works with **daily returns**: one number per trading day.

## Compounding: returns build on each other

If you gain 10% and then another 10%, you've gained 21%, not 20%: the second 10% was earned on a bigger amount. This
is **compounding**. It also works against you: lose 50% and you need a **100%** gain just to get back to where you
started. That lopsidedness is why the app cares so much about avoiding big losses.

**CAGR** (compound annual growth rate) turns a whole price history into one "average yearly growth" figure that
accounts for compounding.

## Volatility: how bumpy is the ride?

**Volatility** measures how much the daily returns jump around. A stock that moves ±0.5% most days has low volatility;
one that often moves ±4% has high volatility. The app reports it **per year** ("annualized") so stocks can be compared
easily. High volatility isn't automatically bad, but it means wider swings in your account, in both directions.

## Drawdown: the worst fall from a high point

A **drawdown** is how far the price has dropped from its previous peak. If a stock reached $200 and then fell to $130,
it was in a 35% drawdown. The **maximum drawdown** is the deepest such fall in the history. For many people this is
the number that matters most emotionally: it is the loss you would have had to sit through.

## The "risk-free" rate, and excess return

You could always lend your money to the U.S. government for a few months by buying **Treasury bills** (T-bills), which
are about as safe as investments get. Their interest rate is called the **risk-free rate**. The app downloads it
automatically (the 13-week T-bill rate); if it can't, it asks you to type one in.

The return a stock earns **above** the risk-free rate is its **excess return**. This is what really matters: if a
risky stock only matches the T-bill rate, you took on risk for nothing. The app's sizing calculations use excess
returns, and any money you don't put in the stock is assumed to earn the risk-free rate.

## Return per unit of risk: the Sharpe ratio

The **Sharpe ratio** divides the average excess return by the volatility. It asks: *how much reward did I get for
each unit of bumpiness?* Higher is better. A Sharpe ratio near zero means the stock barely beat T-bills, once you
account for its swings.

## Lookback window and history

Two sidebar settings control which data is used:

- **History** (5, 10 or 20 years, or "max"): how much price history is loaded.
- **Lookback window** (default 252 trading days, about one year): how much recent history the main sizing estimate
  uses. A shorter window reacts faster but is noisier; a longer one is steadier but slower to notice change.

## Currency

Prices are converted into your chosen **base currency** (U.S. dollars by default) using daily exchange rates, so a
London stock can be compared with a New York one on equal terms. Exchange-rate moves then become part of the return.

## Want to go deeper?

- [Compound interest calculator (investor.gov)](https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator)
- [Treasury bills (TreasuryDirect)](https://www.treasurydirect.gov/marketable-securities/treasury-bills/), the U.S.
  Treasury's own explanation
- [Volatility](https://en.wikipedia.org/wiki/Volatility_(finance)),
  [Drawdown](https://en.wikipedia.org/wiki/Drawdown_(economics)),
  [Sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio),
  [CAGR](https://en.wikipedia.org/wiki/Compound_annual_growth_rate),
  [Stock split](https://en.wikipedia.org/wiki/Stock_split) and [Dividend](https://en.wikipedia.org/wiki/Dividend)
  (Wikipedia)
- The precise conventions the app uses: [Data and conventions](../experienced-investor/01-data-and-conventions.md)
  (experienced track)

[← The big idea](01-the-big-idea.md) · Next: [How much to invest →](03-how-much-to-invest-kelly.md)
