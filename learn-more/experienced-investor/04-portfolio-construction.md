# 4. Portfolio construction

[← Risk and distribution](03-risk-and-distribution.md) · Next: [Backtesting and signals →](05-backtesting-and-signals.md) · Plain-language version: [Building a portfolio](../new-investor/05-building-a-portfolio.md)

## Inputs (`stockapp/portfolio_data.py`, `stockapp/portfolio_input.py`)

- 2–50 tickers, typed or uploaded as CSV (≤ 1 MB). The CSV has a `ticker` column, plus optionally `weight` (current
  holdings, summing to ≤ 1) or `shares` (converted to weights at the latest prices).
- Prices in the base currency are inner-joined on common dates, keeping only dates that have a risk-free rate. At
  least 252 common dates are required.
- **Frequency:** daily, or weekly (W-FRI week-end prices, with rf compounded over the same weeks). Weekly is the
  default when tickers carry different exchange suffixes: asynchronous closes bias daily correlations toward zero.
  Annualization uses 252 or 52.

## Estimation (`stockapp/estimation.py`)

- **μ:** per-asset shrunk mean excess return, the same normal–normal posterior as the single-asset Shrunk Kelly with
  `τ = prior_sd / periods_per_year`, annualized ×P. The sample mean is kept for reference.
- **Σ:** Ledoit–Wolf shrinkage toward a scaled identity
  ([`sklearn.covariance.LedoitWolf`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)),
  annualized ×P. The page reports shrinkage intensity and the condition number before and after. A Cholesky check
  follows; if it fails, the smallest diagonal jitter `ε` on a geometric grid (from 1e-12 × mean variance) is added and
  reported.

Rationale: `Σ⁻¹μ` amplifies estimation error in both inputs. Shrinking μ addresses the dominant error; Ledoit–Wolf
bounds the condition number so the inverse is stable.

## Weightings (`stockapp/optimize.py`, `stockapp/portfolio.py`)

| Weighting | Definition |
|---|---|
| Equal weight | `1/n` |
| HRP | Hierarchical Risk Parity: correlation distance `sqrt(0.5(1 − ρ))`, single-linkage clustering, quasi-diagonal (leaf) order, then recursive bisection with weights inversely proportional to each half's inverse-variance-portfolio variance. Uses no μ. |
| Fractional Kelly, unconstrained | `c·Σ⁻¹μ` (via `numpy.linalg.solve`). May short and lever, so it is shown **for reference only**. |
| Constrained Kelly | The QP below |

**Constrained Kelly QP**

```
maximize    w'μ − (1/(2c)) · w'Σw
subject to  w ≥ 0,  Σ w ≤ gross_max,  w_i ≤ w_max,  Σ_{i ∈ sector s} w_i ≤ cap_s
```

`c` is the sidebar Kelly multiplier: without constraints the optimum is exactly `c·Σ⁻¹μ`, so this is fractional
Kelly with constraints. Defaults: `w_max = 0.25`, `gross_max = 1.0`; sector caps are optional (default 0.40, sectors
from Yahoo). The remainder `1 − Σw` earns rf. The quadratic term is written as `‖sqrt(1/(2c))·L'w‖²` with `Σ = LL'`,
which keeps the problem [DPP-compliant](https://www.cvxpy.org/tutorial/dpp/index.html): it is compiled once, then
re-solved with new parameters (needed for the bootstrap and the frontier). The solver is
[Clarabel](https://clarabel.org/stable/), an interior-point method that meets constraints to about 1e-9.

**Failure handling.** `w = 0` satisfies every constraint, so the problem is always feasible, and with Σ positive
definite the optimum is unique: any failure is numerical. A solution is accepted if its status is `optimal` or
`optimal_inaccurate` **and** every constraint holds within 1e-6. Otherwise the solve is retried once with
`1e-6 × mean(diag Σ)` added to the diagonal. If that fails too, there is **no allocation**: the column shows "—", no
trades are proposed, the split-sample curve and the frontier mark are omitted, and the reason is shown. A substitute
such as equal weight is deliberately never used, because it would read as a recommendation and could violate
`w_max` or the sector caps.

## Uncertainty and evaluation

- **Bootstrap bands** (`stockapp/portfolio_bootstrap.py`): row-wise block bootstrap of the excess-return matrix
  (blocks of 20 daily or 4 weekly periods; resampling whole rows preserves cross-correlation). The app re-estimates μ
  and Σ, re-solves, and reports 5/50/95th percentile weights. It runs up to 200 resamples within a 1.2 s budget,
  timed on a 10-resample pilot (minimum 30). Failed resamples are excluded and counted.
- **Frontier:** 25 values of `c` on a log grid from 0.01 to 10, each reporting `sqrt(w'Σw)` and `w'μ`; equal weight,
  HRP and constrained Kelly are marked.
- **Split-sample check:** every weighting is estimated on the first half of the rows and held fixed (rebalanced each
  period, remainder at rf) over the second half. A full walk-forward test is on the
  [Backtest page](05-backtesting-and-signals.md).
- **Trades** (`stockapp/trades.py`): `target − current` per ticker and cash, in weights and (with a portfolio value)
  shares. Turnover is `½ Σ|Δw|` including cash, and cost is `turnover × bps`. Taxes aren't modelled.

## References

- O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices", *Journal of
  Multivariate Analysis* 88(2), 2004, [doi:10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4);
  the accessible version, ["Honey, I Shrunk the Sample Covariance Matrix" (PDF)](http://www.ledoit.net/honey.pdf)
- M. López de Prado, "Building Diversified Portfolios that Outperform Out of Sample", *Journal of Portfolio
  Management* 42(4), 2016, [doi:10.3905/jpm.2016.42.4.059](https://doi.org/10.3905/jpm.2016.42.4.059) (HRP);
  [Hierarchical Risk Parity (Wikipedia)](https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity)
- [Modern portfolio theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory),
  [Efficient frontier](https://en.wikipedia.org/wiki/Efficient_frontier) and
  [Quadratic programming](https://en.wikipedia.org/wiki/Quadratic_programming) (Wikipedia)

[← Risk and distribution](03-risk-and-distribution.md) · Next: [Backtesting and signals →](05-backtesting-and-signals.md)
