# 1. Data and conventions

[← Track overview](README.md) · Next: [Kelly sizing →](02-kelly-sizing.md) · Plain-language version: [Returns and risk](../new-investor/02-returns-and-risk.md)

## Prices

- **Source:** Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance)
  (`stockapp/providers/yfinance_provider.py`), daily bars with `auto_adjust=False, actions=True`. Analytics use
  **`adj_close`** (split- and dividend-adjusted); Yahoo's `close` is already split-adjusted but not dividend-adjusted.
  FX series use `close`.
- **Dates** are the exchange-local trading date (timezone stripped, normalized). Bars without a close are dropped, e.g.
  an in-progress Asian session.
- **History** loaded per ticker: 5, 10 (default), 20 years or the full history.

## Cache and refresh policy (`stockapp/repository.py`, `stockapp/store.py`)

Bars are cached in SQLite (`data/market.sqlite`, WAL mode):

1. **Cold, or a longer history than cached:** full fetch, replacing the ticker's rows in one transaction.
2. **Fetched within `STOCKAPP_FRESHNESS_HOURS`** (default 6 h): served from cache, no request.
3. **Otherwise incremental:** fetch from 10 cached bars before the last cached date. If `adj_close` on any overlapping
   date (excluding the newest cached bar, which may be mid-session) differs by more than 1e-6 relative, a split or
   dividend has re-based the adjusted series and the **full history is refetched**.
4. If the provider is unavailable or rate-limited and a cache exists, the stale cache is served.

Transient failures (network errors, server errors, rate limiting) get up to 3 attempts in total, with exponential
backoff and jitter. A stale-data warning is shown when the
last bar predates the exchange's previous session.

## Returns

Simple returns `r_t = P_t / P_{t−1} − 1`; log returns `log(1 + r_t)` are also computed. Unless stated otherwise,
estimators use simple returns.

## Risk-free rate (`stockapp/riskfree.py`)

- **Series:** `^IRX`, the 13-week U.S. T-bill yield in annualized percent `y`.
- **Daily rate:** `rf_d = (1 + y/100)^(1/252) − 1`. `^IRX` is a discount-basis yield; treating it as an effective
  annual rate understates the effective rate by about 0.17 pp at a 4% yield (0.05 pp at 2%, 0.26 pp at 5.35%).
- **Alignment:** forward-filled onto the stock's dates for at most 5 observations, **never back-filled** (that would
  leak future rates). Dates with no recent rate are dropped.
- **Fallback:** if `^IRX` can't be loaded, the user enters a constant annual rate.
- **Excess returns:** `e_t = r_t − rf_t`, matched by date.

## Currency (`stockapp/fx.py`)

Prices are converted to the selected base currency: `P_base = P_native / d × FX`. Here `d = 100` for verified
minor-unit quotes (`GBp`, `ZAc`, `ILA`), and `FX` is Yahoo's `XXXBBB=X` close, aligned with the same causal
forward-fill (limit 5) as the risk-free rate. Returns, and hence Kelly, are measured in the base currency, so currency
risk is part of the estimate. The risk-free rate stays the USD T-bill rate whatever the base currency (see
[Limitations](08-limitations.md)).

## Annualization (`stockapp/calendars.py`)

Annualized single-asset metrics use the **number of sessions in the last 365 days on the ticker's exchange**
([`exchange_calendars`](https://github.com/gerrymanoim/exchange_calendars)). Supported suffixes: none (XNYS), `.L`,
`.TO`, `.DE`, `.HK`, `.T`, `.JO`, `.AX`, `.SW`; each was checked against two years of Yahoo bars. Indices, FX/futures
and other suffixes fall back to 252. `.TA` is deliberately excluded: Yahoo has no bars for Tel Aviv's pre-2026 Sunday
sessions. Kelly shrinkage (`τ = prior_sd / 252`) and the portfolio page (252 daily / 52 weekly) use fixed counts.

## Corporate actions

Splits and dividends are stored alongside bars and drawn on the price chart (dashed lines and dots). They aren't used
in calculations beyond the adjustment already in `adj_close`.

## Further reading

- [Risk-free rate](https://en.wikipedia.org/wiki/Risk-free_rate) and
  [Treasury bill](https://en.wikipedia.org/wiki/Treasury_bill) (Wikipedia);
  [Treasury bills (TreasuryDirect)](https://www.treasurydirect.gov/marketable-securities/treasury-bills/)
- [Stock split](https://en.wikipedia.org/wiki/Stock_split) and [Exchange rate](https://en.wikipedia.org/wiki/Exchange_rate)
  (Wikipedia)

[← Track overview](README.md) · Next: [Kelly sizing →](02-kelly-sizing.md)
