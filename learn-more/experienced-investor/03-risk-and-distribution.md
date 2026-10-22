# 3. Risk and distribution diagnostics

[← Kelly sizing](02-kelly-sizing.md) · Next: [Portfolio construction →](04-portfolio-construction.md) · Plain-language version: [Reading the risk tabs](../new-investor/04-reading-the-risk-tabs.md)

These tabs exist mainly to test the assumptions behind the Kelly estimates: normality, independence, stationarity and
a single regime. Risk and distribution metrics use the lookback window; regimes, factors and relative performance use
the full loaded history. `P` below is the exchange's sessions per year.

## Risk (`stockapp/risk.py`)

| Metric | Definition |
|---|---|
| Historical VaR 95% | `−quantile(r, 0.05)` |
| Historical CVaR 95% | `−mean(r \| r ≤ quantile(r, 0.05))` |
| Parametric VaR 95% | `−(mean(r) + sd(r)·Φ⁻¹(0.05))`. The gap to historical VaR is a quick fat-tail read. |
| Max drawdown | `max_t (1 − P_t / max_{s≤t} P_s)`, with peak, trough, recovery dates and duration (calendar days) |
| Sortino | `mean(e) / sqrt(mean(min(e, 0)²)) · √P`: downside deviation against zero excess return |
| Calmar | `CAGR / max drawdown` over the lookback |
| Omega (threshold 0) | `Σ max(r, 0) / Σ max(−r, 0)` |
| Beta, alpha, R² | OLS `r_t = a + β·b_t + ε_t` on common dates with the benchmark's returns; alpha annualized `a·P` |

**Volatility:** 21-day rolling SD × √P, and RiskMetrics-style EWMA with λ = 0.94: zero mean,
`σ²_t = λσ²_{t−1} + (1 − λ)r_t²`, annualized. The value at `t` uses returns through `t`, so it is the forecast for
`t + 1`. **GARCH(1,1):** constant mean, normal innovations (the [`arch`](https://arch.readthedocs.io/en/latest/univariate/introduction.html)
package), fitted on percent returns for numerical stability. The analytic variance forecast for days 1–21 is averaged
and annualized. If the fit fails or doesn't converge, the page falls back to EWMA and says so.

## Distribution (`stockapp/distribution.py`)

| Diagnostic | Definition |
|---|---|
| Skewness, excess kurtosis | Moment estimators `m₃/m₂^1.5` and `m₄/m₂² − 3` (population central moments) |
| Jarque–Bera | `JB = n/6·(S² + K²/4)`; p-value `exp(−JB/2)` (closed-form χ²₂ survival function) |
| Ljung–Box (lag 10) | `Q = n(n+2) Σ_{k=1..10} ρ_k² / (n−k)`, p-value from χ²₁₀. On squared returns this is the McLeod–Li test for volatility clustering. |
| ACF | Lags 1–20 for returns and squared returns, with ±1.96/√n bands |
| Q-Q | Against N(0, 1) (standardized returns) and against a maximum-likelihood Student-t, at plotting positions `(i − 0.5)/n` |
| Hurst exponent | Rescaled range: windows from 8 doubling to n/2, slope of log(R/S) on log(size). Labelled indicative: biased upward and noisy on short samples. |

**Verdict ("Is Gaussian Kelly safe?")** flags *fat tails* if excess kurtosis > 1, *non-normal* if JB p < 0.01, and
*volatility clustering* if Ljung–Box on squared returns has p < 0.01. Any flag recommends the Shrunk or Log-optimal
estimate, since the mean–variance `f*` ignores tail losses that the log objective penalizes.

## Regimes (`stockapp/regimes.py`)

**Causal volatility regimes:** the 21-day rolling SD at `t` is compared with the 33rd and 67th percentiles of **all
rolling-SD values up to and including `t`** (expanding window, at least 252 values), giving low / normal / high. No
label depends on data after its date, and appending data never changes past labels. A label at `t` includes `r_t`, so
a rule acting at `t` uses it with the usual one-day execution lag. The table reports Gaussian and shrunk `f*` per
regime and warns below 60 days in a regime.

**HMM view:** a Gaussian HMM ([`hmmlearn`](https://hmmlearn.readthedocs.io/en/latest/), 2 or 3 states, diagonal
covariance, states ordered by variance), decoded with Viterbi on the full sample. Every label therefore depends on
future data. hmmlearn exposes no filtered (forward-only) probabilities, so the view is shown as **historical only** and
is never used by any signal, feature or sizing rule.

## Factor regression (`stockapp/factors.py`)

- **Data:** the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
  daily Fama–French 5 factors (2×3) and momentum, downloaded over HTTPS with size and zip-entry limits. The parser
  checks the published layout and units, and the data is cached in SQLite and refreshed at most weekly.
- **Model:** OLS of the stock's return minus the *library's own* RF on Mkt-RF, SMB, HML, RMW, CMA and Mom plus an
  intercept, over common dates (at least 60).
- **Inference:** Newey–West HAC standard errors, Bartlett kernel, lag 5, no small-sample correction.
- **Summary:** R² ≥ 0.6 "mostly market beta", ≥ 0.3 "partly explained", otherwise "explains little". Loadings with
  |t| > 2 are named as tilts. U.S. factors fit non-U.S. listings poorly, and the page warns.

## Relative performance and seasonality (`stockapp/relative.py`, `stockapp/seasonality.py`)

Trailing 21/63/126/252-session returns against the benchmark and the stock's SPDR sector ETF (Yahoo sector → XLK, XLF,
XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC), plus rolling returns relative to the benchmark over the same windows. Seasonality is a
month × year table of month-end-to-month-end returns; it is shown for context and is mostly noise for single stocks.

## References

- T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity", *Journal of Econometrics* 31(3), 1986,
  [doi:10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1)
- [RiskMetrics Technical Document (J.P. Morgan/Reuters, 1996; PDF hosted by MSCI)](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a):
  the EWMA λ = 0.94 convention
- W. K. Newey and K. D. West, "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix", *Econometrica* 55(3), 1987, [doi:10.2307/1913610](https://doi.org/10.2307/1913610)
- E. F. Fama and K. R. French, "A five-factor asset pricing model", *Journal of Financial Economics* 116(1), 2015,
  [doi:10.1016/j.jfineco.2014.10.010](https://doi.org/10.1016/j.jfineco.2014.10.010); the
  [5-factor data description](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html)
- Wikipedia: [Value at risk](https://en.wikipedia.org/wiki/Value_at_risk),
  [Expected shortfall](https://en.wikipedia.org/wiki/Expected_shortfall),
  [Jarque–Bera test](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test),
  [Ljung–Box test](https://en.wikipedia.org/wiki/Ljung%E2%80%93Box_test),
  [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent),
  [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution),
  [Hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model),
  [Newey–West estimator](https://en.wikipedia.org/wiki/Newey%E2%80%93West_estimator)

[← Kelly sizing](02-kelly-sizing.md) · Next: [Portfolio construction →](04-portfolio-construction.md)
