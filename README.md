# Fractional Kelly Stock Analyzer (Streamlit)

A simple Streamlit app that fetches free market data using `yfinance`, computes fractional Kelly allocations from historical daily excess returns (over the T-bill rate) with uncertainty bands, and visualizes price history, return distribution, cumulative returns, and Kelly across lookback windows. Users choose the lookback window, Kelly fraction, estimation method, shrinkage prior and drawdown limit.

## Features
- Free data source via yfinance (no API key); split/dividend-adjusted daily closes
- Price history cached per ticker, so slider changes are instant
- Adjustable lookback (60–1000 trading days) and Kelly fraction (0.1–1.0)
- Daily returns and log returns calculation
- Risk-free rate from the 13-week US T-bill (`^IRX`), with a manual fallback if it can't be fetched
- Four Kelly estimates on excess returns (see [Kelly methods](#kelly-methods)); applied fraction = `clip(fraction × f*, 0, 1)` (no shorting or leverage)
- Bootstrap 5–95% bands on Kelly, with a warning when the data can't tell the edge from zero
- Annualized metrics: CAGR, volatility, Sharpe ratio, and expected growth at the applied fraction
- Charts: price history, return histogram, cumulative returns, Kelly across windows with error bars
- Ticker input validation and friendly errors for unknown tickers or network failures
- CSV download of returns, daily risk-free rate and excess returns

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

- `src/app.py`: Streamlit entry point
- `src/stockapp/`: installable package (`data`, `kelly`, `charts`)
- `tests/`: pytest suite; `tests/fixtures.py` generates synthetic prices

## Secrets

The app needs no secrets today. If a later feature does, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored) and fill it in locally.

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

## Notes
- Kelly is sensitive to estimation error even with shrinkage; use conservative fractions.
- `^IRX` is a discount-basis yield; treating it as an effective annual rate understates the rate slightly (about 0.17 percentage points at 4%). The rate is USD-only, so non-USD tickers should really use a local rate (not yet supported).
- Annualized growth at the applied fraction uses the second-order approximation `g(f) ≈ 252·[mean(rf) + f·mean(e) − f²σ²/2]`.
- Five years of adjusted daily closes are fetched once per ticker and cached in memory for one hour; the lookback slider (in trading days) slices the cached data.
- The "Kelly across windows" chart uses fixed windows (60/120/252/504 trading days) of the full history, independent of the lookback slider.
- This is for educational purposes and not financial advice.
