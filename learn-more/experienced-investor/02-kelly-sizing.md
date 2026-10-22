# 2. Kelly sizing

[← Data and conventions](01-data-and-conventions.md) · Next: [Risk and distribution →](03-risk-and-distribution.md) · Plain-language version: [How much to invest](../new-investor/03-how-much-to-invest-kelly.md)

## Objective

For a fraction `f` held in the stock and `1 − f` at the risk-free rate, the daily portfolio return is
`R_f = rf + f·e`, where `e = r − rf`. Kelly chooses `f` to maximize the expected log growth `E[log(1 + R_f)]`, which
maximizes the long-run compound growth rate. A second-order expansion gives

```
g(f) ≈ mean(rf) + f·μ_e − f²·σ_e²/2      →      f* = μ_e / σ_e²
```

the continuous-time Kelly `(μ − r)/σ²`. The Research page reports annualized growth at the applied fraction with
this approximation: `g_ann(f) = P·[mean(rf) + f·mean(e) − f²·var(e)/2]`, with `P` the exchange's sessions per year.

**Applied fraction** (all methods): `clip(k · f*, 0, 1)`, where `k` is the sidebar multiplier (0.1–1.0, default
0.5). No shorting, no leverage. The raw `f*` is shown unclamped as a diagnostic.

Why fractional: `f*` is proportional to `μ_e`, whose standard error `σ/√n` is often larger than `μ_e` itself over a
one-year window. Growth is asymmetric around the optimum: in the quadratic approximation, betting `2f*` gives zero
excess growth, while `0.5f*` keeps 75% of it at half the volatility. Overestimating the edge is costly, while
underestimating it costs little.

## The four estimators (`stockapp/kelly.py`, `stockapp/drawdown.py`)

All use daily excess returns over the lookback window (default 252 trading days, minimum 60).

| Method | Estimator | Notes |
|---|---|---|
| **Gaussian** | `f* = mean(e) / var(e)` (sample variance) | 0 if `var ≤ 1e-12`. Treats the sample mean as the true mean. |
| **Shrunk** (default) | `f* = μ_post / var(e)` with `μ_post = mean(e)·τ² / (τ² + var(e)/n)`, `τ = prior_sd / 252` | Normal–normal posterior mean with a zero-mean prior on the *daily* expected excess return; `prior_sd` is the annual prior SD (sidebar, 0.01–0.50, default 0.10). Shrinks in proportion to the noise ratio `se²/τ²`. |
| **Log-optimal** | `argmax_{f ∈ [0, 10]} mean(log(1 + rf + f·e))` on a 0.005 grid | Empirical, distribution-free. Any `f` that makes `1 + R_f ≤ 0` on some day is infeasible. `1 + rf + f·e` falls linearly in `f` on every day with `e < 0`, so the feasible set is `[0, f_crit)` and the search stops at the first fully infeasible block of the grid. The Gaussian–log-optimal gap is the fat-tail penalty. |
| **Drawdown-constrained** | Largest `f ∈ {0, 0.05, …, 1}` whose 90th-percentile max drawdown over 252 days is ≤ the limit (sidebar 10–50%, default 25%) | 1,000 block-bootstrap paths (20-day blocks, seed 42) of the observed `(r, rf)` pairs. Common random numbers across `f`. Equity `cumprod(1 + rf + f·e)`, running peak starting at 1. Returns 0 if no grid point qualifies. |

## Bootstrap bands (`stockapp/bootstrap.py`)

Daily returns aren't i.i.d.: volatility clusters, and there is short-range autocorrelation. Bands therefore use a
**moving-block bootstrap**: 1,000 resamples of the lookback's excess returns, built from contiguous 20-day blocks with
uniformly drawn starts (seed 42, so the same data always gives the same band). On each resample the app recomputes
Gaussian `f*` and shrunk `f*` (same shrinkage formula, using the resample's variance), and reports the 5th, 50th and
95th percentiles and `P(f* > 0)`.

If the Gaussian band's 5–95% range contains 0, the page warns that the data can't distinguish the expected excess
return from zero. The **shrunk** band's 5th percentile is an input to the advice rule ([page 7](07-advice-rules.md)).

## Kelly across windows

The windowed chart recomputes the selected method's applied fraction on the last 60, 120, 252 and 504 sessions of the
full history (independent of the lookback slider), with bootstrap error bars for the Gaussian and Shrunk methods.
Dispersion across windows is a direct read on parameter instability.

## References

- J. L. Kelly Jr., "A New Interpretation of Information Rate", *Bell System Technical Journal* 35(4), 1956,
  [doi:10.1002/j.1538-7305.1956.tb03809.x](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x)
  ([PDF mirror](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf))
- [Kelly criterion (Wikipedia)](https://en.wikipedia.org/wiki/Kelly_criterion): fractional Kelly and the growth/risk
  trade-off
- [Shrinkage (statistics)](https://en.wikipedia.org/wiki/Shrinkage_(statistics)),
  [Prior probability](https://en.wikipedia.org/wiki/Prior_probability) and
  [Bootstrapping, including block bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) (Wikipedia)

[← Data and conventions](01-data-and-conventions.md) · Next: [Risk and distribution →](03-risk-and-distribution.md)
