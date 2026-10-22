# Fractional Kelly Stock Analyzer (Streamlit)

A Streamlit app that sizes stock positions with the Kelly criterion, using free market data from `yfinance`. It
estimates fractional Kelly allocations from daily excess returns over the T-bill rate, with bootstrap uncertainty bands,
and checks the assumptions behind them with risk, distribution, regime and factor analytics. It also builds
constrained multi-asset portfolios, tests trading rules walk-forward, and turns a calibrated probability model plus
explicit rules into BUY / SELL / HOLD suggestions, with each rule's out-of-sample track record shown beside it.

**Educational tool, not investment advice.** Past performance does not predict future results.

## Learn more

The [`learn-more/`](learn-more/README.md) guides explain why the app uses the algorithms it does and how it reaches
its numbers and BUY / SELL / HOLD suggestions, in two linked tracks:

- **[New investor](learn-more/new-investor/README.md)**: plain language, no formulas. Covers
  [returns and risk](learn-more/new-investor/02-returns-and-risk.md),
  [the Kelly idea](learn-more/new-investor/03-how-much-to-invest-kelly.md),
  [portfolios](learn-more/new-investor/05-building-a-portfolio.md),
  [backtests](learn-more/new-investor/06-testing-ideas-on-the-past.md),
  [the model and the advice](learn-more/new-investor/07-the-model-and-the-advice.md), and a
  [glossary](learn-more/new-investor/glossary.md).
- **[Experienced investor](learn-more/experienced-investor/README.md)**: formulas, parameters, design trade-offs and
  references. Covers [data conventions](learn-more/experienced-investor/01-data-and-conventions.md),
  [Kelly sizing](learn-more/experienced-investor/02-kelly-sizing.md),
  [risk diagnostics](learn-more/experienced-investor/03-risk-and-distribution.md),
  [portfolio construction](learn-more/experienced-investor/04-portfolio-construction.md),
  [backtesting](learn-more/experienced-investor/05-backtesting-and-signals.md),
  [the probability model](learn-more/experienced-investor/06-probability-model.md),
  [advice rules](learn-more/experienced-investor/07-advice-rules.md) and
  [limitations](learn-more/experienced-investor/08-limitations.md).

## Features

The sidebar settings are shared by every page: ticker, history (5, 10 or 20 years, or max), base currency, benchmark,
Kelly fraction (0.1–1.0), Kelly method, prior SD and max tolerable drawdown.

**Research** (one ticker)
- Free data via yfinance (no API key): split- and dividend-adjusted daily closes, converted to the base currency (USD,
  EUR, GBP, JPY, CAD, CHF, AUD or HKD) with daily FX rates
- Risk-free rate from the 13-week US T-bill (`^IRX`), with a manual fallback if it can't be fetched
- Four Kelly estimates on excess returns (see [Kelly methods](#kelly-methods)); applied fraction
  `clip(fraction × f*, 0, 1)`, with no shorting or leverage. The lookback window runs from 60 trading days to the
  length of the loaded history.
- Bootstrap 5–95% bands on Kelly, with a warning when the data can't tell the edge from zero, and Kelly across fixed
  windows
- CAGR, volatility, Sharpe ratio and approximate growth at the applied fraction, annualized with the ticker's own
  exchange calendar
- Tabs: price (with splits and dividends), returns, risk (VaR / CVaR, drawdown, Sortino, Calmar, Omega, beta, EWMA and
  GARCH volatility), distribution tests ("Is Gaussian Kelly safe?"), causal volatility regimes, a Fama–French 5-factor
  and momentum regression, and performance relative to the benchmark and sector ETF with seasonality
- CSV download of returns, and an offline HTML research report (print it to PDF from a browser)

**Portfolio** (2–50 tickers, typed or uploaded as a CSV of up to 1 MB with optional current weights or share counts)
- Shrunk expected returns and Ledoit–Wolf covariance; daily or weekly returns (weekly by default across exchanges)
- Equal weight, HRP, an unconstrained Kelly reference, and constrained Kelly (long-only, per-asset, total-invested
  and optional sector caps)
- Bootstrap bands on the weights, efficient frontier, correlation clustering, and a split-sample test
- Trade list from current to target weights with turnover and cost. If the optimizer fails, no allocation or trades are
  shown rather than a substitute.

**Backtest & Signals**
- Walk-forward backtests with transaction costs and slippage, against buy & hold and Kelly buy & hold; trial counts
  are shown
- Signal library (SMA cross, Donchian, momentum 12-1, RSI, Bollinger, volatility targeting, regime filter), with
  out-of-sample cards for every signal
- Calibrated LightGBM probability of beating the T-bill rate, scored against climatology
- BUY / SELL / HOLD advice from explicit rules. The model's probability is used only once it has passed the skill
  gate, and the rule's own walk-forward track record and a one-year outcome fan are shown alongside.

**Settings, data and operations**
- Optional OIDC sign-in for saved watchlists and settings (see [Secrets and sign-in](#secrets-and-sign-in))
- SQLite price cache with incremental updates and restatement detection; stale-data warnings
- Structured JSON logs; an optional diagnostics panel (see [Configuration](#configuration))
- [Watchlist scanner](#watchlist-scanner) CLI with change alerts by email, and a [Docker image](#docker)

## Quickstart

Requires Python 3.12+.

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

2. Install the pinned dependencies, then the app itself in editable mode with dev tools:

```bash
pip install -r requirements.lock
pip install -e ".[dev]"
```

3. Run the app:

```bash
streamlit run src/app.py
```

Open the provided local URL in your browser.

4. Optional: install the git hooks so ruff runs on every commit:

```bash
pre-commit install
```

## Checks

With dependencies installed (step 2 above), run the same checks as CI:

```bash
ruff check              # lint (add --fix to auto-fix)
ruff format --check     # formatting (drop --check to reformat)
mypy src/stockapp       # strict type checking
pytest                  # tests; network access is blocked, so they run offline
pip-audit --skip-editable  # known vulnerabilities in installed packages
```

CI (`.github/workflows/ci.yml`) runs these on every push and pull request for Python 3.12 and 3.13.

## Project layout

- `src/app.py`: Streamlit entry point (page config, shared sidebar, navigation)
- `src/app_pages/`: one thin script per page, registered in `src/stockapp_ui/router.py`
- `src/stockapp/`: the analytics package, with no Streamlit dependency. It holds the data layer (`providers/`,
  `repository`, `store`), Kelly and risk analytics, `portfolio*` and `optimize`, `backtest/` and `signals/`, the
  probability model (`model/`), `advice`, the `report` export and the `scan` CLI.
- `src/stockapp_ui/`: the Streamlit layer: shared state, cached loaders, sign-in, diagnostics and the page views
  (`views/`)
- `tests/`: pytest suite; `tests/fixtures.py` generates synthetic prices
- `data/`: SQLite caches (`market.sqlite` for prices and factors, `users.sqlite` for saved user data); git-ignored
- `learn-more/`: documentation on the algorithms and decision strategy, for
  [new](learn-more/new-investor/README.md) and [experienced](learn-more/experienced-investor/README.md) investors

## Configuration

Runtime settings come from environment variables; none of them are secrets.

| Variable | Default | Effect |
|---|---|---|
| `STOCKAPP_DB_PATH` | `data/market.sqlite` | Price cache location; relative paths resolve against the project root, and the path must be inside `data/` |
| `STOCKAPP_FRESHNESS_HOURS` | `6` | Cached bars younger than this are served without contacting the provider |
| `LOG_LEVEL` | `INFO` | Level for the JSON logs on stdout |
| `DEBUG_UI` | unset | `1` adds a diagnostics panel to the Settings page (app version, data-layer cache hits, recent fetches) |

Logs are JSON lines with `ts`, `level`, `logger`, `event`, `duration_ms`, `ticker` and `user` (the first 8 characters
of the hashed user key). Fields whose names look secret are redacted, and exceptions are logged by type only.

`.streamlit/config.toml` hardens the server: XSRF and CORS protection on, a 1 MB upload cap, error details hidden
from users, the developer toolbar hidden, and usage statistics off.

## Docker

The `Dockerfile` builds a slim runtime image with the locked runtime dependencies only (no dev tools or tests). It runs
as a non-root user, keeps the application files read-only, and has a health check on Streamlit's `/_stcore/health`
endpoint.

```bash
docker build -t stockapp .
docker run -p 8501:8501 -v stockapp-data:/app/data stockapp
```

`/app/data` holds the SQLite caches and saved user data, so give it a volume to keep them across containers. Secrets
aren't copied into the image (`.dockerignore` excludes them). To enable sign-in, mount the secrets file read-only, and
set `redirect_uri` to the URL users reach the container on:

```bash
docker run -p 8501:8501 -v stockapp-data:/app/data \
  -v "$PWD/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro" stockapp
```

The [Configuration](#configuration) variables can be passed with `-e`, for example `-e LOG_LEVEL=DEBUG`.

## Secrets and sign-in

The app needs no secrets: without any, it runs anonymously with every analytics feature. Optional sign-in adds saved
watchlists and settings. Secrets come only from `.streamlit/secrets.toml` (git-ignored) or the environment; start from
`.streamlit/secrets.toml.example`.

### Configuring sign-in locally

Sign-in uses Streamlit's native OpenID Connect support (`st.login`), which needs the `streamlit[auth]` extra (Authlib),
installed with the app. Any OIDC provider works (Google, Microsoft Entra ID, Auth0, Okta, ...); embedded apps aren't
supported.

1. Register an OAuth / OIDC web application with your provider and add the redirect URI
   `http://localhost:8501/oauth2callback` (use your deployed URL in production; it must end in `/oauth2callback`).
2. Generate a strong cookie secret and an app salt:

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"   # cookie_secret
   python -c "import secrets; print(secrets.token_urlsafe(32))"   # app_salt
   ```

3. Fill in `.streamlit/secrets.toml`:

   ```toml
   app_salt = "..."

   [auth]
   redirect_uri = "http://localhost:8501/oauth2callback"
   cookie_secret = "..."
   client_id = "..."
   client_secret = "..."
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```

4. Restart the app; a sign-in button appears in the sidebar.

Saved data is keyed by `HMAC-SHA256(app_salt, [iss, sub])`, where `iss` identifies the provider and `sub` is its stable
subject claim for the user; no email, name or raw claim is stored. Including `iss` means users of different providers
can never share a key, so switching or adding providers is safe. Keep `app_salt` constant, since changing it orphans
existing saved data. The Settings page has a "delete my data" button.

Earlier versions keyed data by `sha256(app_salt + sub)`. A user's data is moved to the new key automatically the next
time they sign in. That old key has no `iss`, so don't switch providers until existing users have signed in once:
after a switch, a different person with the same `sub` could otherwise sign in first and receive that data.

## Dependencies

`pyproject.toml` is the single source of truth: edit version ranges there (runtime deps under `[project] dependencies`, dev tools under `[project.optional-dependencies] dev`). `requirements.lock` pins every package, dev tools included, and is generated from it with pip-tools. CI installs from the lockfile. After changing `pyproject.toml`, regenerate and commit the lockfile:

```bash
pip-compile --extra dev --strip-extras --allow-unsafe --no-emit-index-url -o requirements.lock pyproject.toml
```

To upgrade everything within the allowed ranges, add `--upgrade`.

## Kelly methods

All methods use daily excess returns `e = r − rf`, where `r` is the stock's simple return and `rf` the daily T-bill rate, `(1 + y/100)^(1/252) − 1` for an annual yield `y`%. The applied fraction is always `clip(fraction × f*, 0, 1)`.

| Method | Formula | What it assumes |
|---|---|---|
| Gaussian | `f* = mean(e) / var(e)`, the continuous-time Kelly `(μ − r) / σ²` | Normal returns; the sample mean is the true mean |
| Shrunk (default) | `f* = μ_post / var(e)`, with `μ_post = mean(e) · τ² / (τ² + var(e)/n)` and `τ = prior SD / 252` | A normal prior on the annual expected excess return, centred on 0 with the SD you choose |
| Log-optimal | `f* = argmax_f mean(log(1 + rf + f·e))` over `f ∈ [0, 10]` | Only that the past sample is representative; no normality, so fat tails lower `f*` |
| Drawdown-constrained | Largest `f ∈ {0, 0.05, …, 1}` whose 90th-percentile max drawdown over 1 year is within your limit | Bootstrapped one-year paths of the observed returns |

Shrinkage matters because the daily mean is very noisy: its standard error `σ/√n` is often larger than the mean itself, so the Gaussian `f*` can swing wildly. The shrunk estimate pulls the mean toward 0 in proportion to that noise.

**Bootstrap bands.** The app resamples the lookback's excess returns 1,000 times using 20-day blocks (a moving-block bootstrap, which keeps volatility clustering and short-range autocorrelation within each block) and recomputes Gaussian and Shrunk Kelly on each resample. The table shows the 5th–95th percentile range: roughly, the values of `f*` that are consistent with the data. If that band includes 0, the data can't distinguish the stock's expected excess return from zero and the app says so; Kelly sizing is unreliable there.

For the reasoning behind each method, see [Kelly sizing](learn-more/experienced-investor/02-kelly-sizing.md), or the plain-language [How much to invest: the Kelly idea](learn-more/new-investor/03-how-much-to-invest-kelly.md).

## Watchlist scanner

`python -m stockapp.scan` runs the Advice tab's BUY / SELL / HOLD rules for every ticker in a watchlist CSV (a
`ticker` column, optionally `weight` for current holdings) through the cached data layer, and writes an HTML report
plus a JSON file of actions next to it:

```bash
python -m stockapp.scan --watchlist watchlist.csv --out reports/scan.html            # also writes reports/scan.json
python -m stockapp.scan --watchlist watchlist.csv --out reports/scan.html --email    # email only changed actions
```

`--email` compares with the previous JSON and sends only tickers whose action changed, over SMTP with STARTTLS. It
reads `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS` and `ALERT_TO` from the environment; nothing is stored in
the repo. Each ticker trains the walk-forward model, so expect several seconds per ticker.

Prices are in USD. `--horizon` (21 or 5 trading days, default 21), `--years` (history, default 10) and `--lookback`
(Kelly window, default 252) are adjustable. The Kelly fraction (0.5), prior SD (0.10) and max drawdown (25%) are fixed
at the app's defaults. The scanner uses the same skill gate as the app: without demonstrated skill, p = 0.5.

### Scheduling

The scanner doesn't schedule itself. With cron (weekdays at 22:30 UTC, after the US close):

```cron
30 22 * * 1-5  cd /path/to/stock-ticker-demo && .venv/bin/python -m stockapp.scan --watchlist watchlist.csv --out reports/scan.html --email >> reports/scan.log 2>&1
```

Or as a scheduled GitHub Actions workflow, with the SMTP settings stored as repository secrets. Keep the previous
`reports/scan.json` between runs (for example with `actions/cache`) so `--email` can detect changes:

```yaml
on:
  schedule:
    - cron: "30 22 * * 1-5"
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with: {python-version: "3.12", cache: pip}
      - run: pip install -r requirements.lock && pip install -e .
      - run: python -m stockapp.scan --watchlist watchlist.csv --out reports/scan.html --email
        env:
          SMTP_HOST: ${{ secrets.SMTP_HOST }}
          SMTP_PORT: ${{ secrets.SMTP_PORT }}
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASS: ${{ secrets.SMTP_PASS }}
          ALERT_TO: ${{ secrets.ALERT_TO }}
```

## Notes
- Kelly is sensitive to estimation error even with shrinkage; use conservative fractions.
- `^IRX` is a discount-basis yield; treating it as an effective annual rate understates the rate slightly (about 0.17 percentage points at 4%). The rate is USD-only, so non-USD tickers should really use a local rate (not yet supported).
- Annualized growth at the applied fraction uses the second-order approximation `g(f) ≈ P·[mean(rf) + f·mean(e) − f²σ²/2]`, where `P` is the number of trading sessions in the last year on the ticker's exchange (252 when the exchange's calendar isn't known). CAGR, volatility and the Sharpe ratio are annualized with the same `P`.
- Price history (10 years by default) is stored in the SQLite cache and refreshed incrementally once it is older than `STOCKAPP_FRESHNESS_HOURS`; an in-memory layer shared by every session of the server caches it for 5 minutes. The lookback slider (in trading days) slices the loaded data, so changing it doesn't refetch.
- The "Kelly across windows" chart uses fixed windows (60/120/252/504 trading days) of the full history, independent of the lookback slider.
- Survivorship bias: analyses use currently listed tickers; delisted companies are absent, which biases historical comparisons upward.
- This is for educational purposes and not financial advice.
