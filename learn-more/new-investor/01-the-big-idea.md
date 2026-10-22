# 1. The big idea

[← Track overview](README.md) · [Glossary](glossary.md) · Next: [Returns and risk →](02-returns-and-risk.md)

## What the app does, in one paragraph

You type in a stock's **ticker**, its short code on the stock exchange (for example `MSFT` for Microsoft). The app
downloads that stock's price history, measures how it has behaved, and answers one main question: *if this stock keeps
behaving roughly the way it has, what share of my money would it be sensible to put in it?* Everything else in the app
helps you judge how much to trust that answer.

## Why "how much" matters as much as "what"

Most people think investing is about picking the right stock. Picking matters, but **how much** you put in matters
just as much, and it gets far less attention.

Here is a simple way to see it. Imagine a game where you flip a coin that lands heads 60% of the time. Heads, you win
whatever you bet; tails, you lose it. The odds are in your favour, so you should play. But how much should you bet
each time?

- **Bet everything every time:** sooner or later a tails comes up and you lose it all. One bad flip ends the game,
  however good the odds were.
- **Bet a tiny amount:** you'll almost never get hurt, but your money grows very slowly.
- **Somewhere in between** is a bet size that makes your money grow fastest over many flips, without the risk of
  wiping yourself out.

That "best in between" bet is what the **Kelly criterion** finds. It is the core idea of this app, and
[page 3](03-how-much-to-invest-kelly.md) explains it properly. The key lesson already shows: a good opportunity, sized
badly, can still ruin you.

## Why the app is careful, even cautious

Real stocks are not coins. We don't know their true odds; we only see their past, and the past is noisy. So the app is
built around being honest about uncertainty:

- It **scales the Kelly answer down** (by default to half), because the full answer assumes we know the odds exactly.
- It shows **ranges, not just single numbers**, so you can see how sure (or unsure) an estimate is.
- It **never suggests borrowing money or betting against a stock**. Every suggested position is between 0% and 100%
  of your money.
- It **tests its own rules on past data** in a fair way, and shows you the results even when they're unimpressive.
- When its prediction model has **no proven skill**, it says so and ignores the prediction.

## The four pages of the app

| Page | What it's for |
|---|---|
| **Research** | One stock in depth: price history, how much to invest, risk, how unusual its returns are, its behaviour in calm vs stormy markets, and comparisons with the wider market |
| **Portfolio** | Several stocks together: how to split money between them, and what trades would get you from where you are to the suggested mix |
| **Backtest & Signals** | Tests trading rules on past data, a probability model, and the BUY / SELL / HOLD suggestion |
| **Settings** | Reset the sidebar, and (if sign-in is set up) save watchlists and settings |

The sidebar settings (ticker, how many years of history, currency, the Kelly multiplier and so on) are shared by every
page.

## Want to go deeper?

- The U.S. Securities and Exchange Commission's beginner site:
  [Introduction to investing (investor.gov)](https://www.investor.gov/introduction-investing)
- [Kelly criterion (Wikipedia)](https://en.wikipedia.org/wiki/Kelly_criterion), including the coin-flip example
- The formulas behind this page: [Kelly sizing](../experienced-investor/02-kelly-sizing.md) (experienced track)

[← Track overview](README.md) · Next: [Returns and risk →](02-returns-and-risk.md)
