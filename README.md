# Fractional Kelly Stock Analyzer (Streamlit)

A simple Streamlit app that fetches free market data using `yfinance`, computes a fractional Kelly allocation from historical daily returns, and visualizes price history, return distribution, cumulative returns, and Kelly across lookback windows. Users can adjust the lookback window and Kelly fraction via sliders.

## Features
- Free data source via yfinance (no API key); split/dividend-adjusted daily closes
- Price history cached per ticker, so slider changes are instant
- Adjustable lookback (60–1000 trading days) and Kelly fraction (0.1–1.0)
- Daily returns and log returns calculation
- Fractional Kelly: `f = fraction * (mu / var)`, clamped to [0, 1] (no shorting or leverage)
- Charts: price history, return histogram, cumulative returns, Kelly across windows
- Ticker input validation and friendly errors for unknown tickers or network failures
- CSV download of computed returns

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

## Notes
- This app uses historical daily returns to estimate mean/variance; Kelly is sensitive to estimation error—use conservative fractions.
- Five years of adjusted daily closes are fetched once per ticker and cached in memory for one hour; the lookback slider (in trading days) slices the cached data.
- The "Kelly across windows" chart uses fixed windows (60/120/252/504 trading days) of the full history, independent of the lookback slider.
- This is for educational purposes and not financial advice.
