# Fractional Kelly Stock Analyzer (Streamlit)

A simple Streamlit app that fetches free market data using `yfinance`, computes a fractional Kelly allocation from historical daily returns, and visualizes price history, return distribution, cumulative returns, and Kelly across lookback windows. Users can adjust the lookback window and Kelly fraction via sliders.

## Features
- Free data source via yfinance (no API key)
- Daily returns and log returns calculation
- Fractional Kelly: `f = fraction * (mu / var)` with safety clamps
- Charts: price history, return histogram, cumulative returns, Kelly across windows
- CSV download of computed returns

## Quickstart

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
.venv\\Scripts\\activate
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

## Notes
- This app uses historical daily returns to estimate mean/variance; Kelly is sensitive to estimation error—use conservative fractions.
- No offline caching is implemented; data is fetched on demand.
- This is for educational purposes and not financial advice.
