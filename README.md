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

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run src/app.py
```

Open the provided local URL in your browser.

## Tests

With dependencies installed (step 2 above):

```bash
pytest
```

## Notes
- This app uses historical daily returns to estimate mean/variance; Kelly is sensitive to estimation error—use conservative fractions.
- Five years of adjusted daily closes are fetched once per ticker and cached in memory for one hour; the lookback slider (in trading days) slices the cached data.
- The "Kelly across windows" chart uses fixed windows (60/120/252/504 trading days) of the full history, independent of the lookback slider.
- This is for educational purposes and not financial advice.
