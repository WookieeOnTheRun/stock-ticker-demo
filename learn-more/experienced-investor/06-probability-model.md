# 6. The probability model

[← Backtesting and signals](05-backtesting-and-signals.md) · Next: [Advice rules →](07-advice-rules.md) · Plain-language version: [The model and the advice](../new-investor/07-the-model-and-the-advice.md)

**Target:** `P(stock beats the T-bill rate over the next H sessions)`, with `H ∈ {21, 5}` (default 21). The design
aims at an honest probability rather than a strong-looking one: purged cross-validation, held-out calibration,
walk-forward scoring, and a hard comparison against climatology.

## Features (`stockapp/model/features.py`)

Every feature at `t` uses prices (and benchmark prices) through `t` only. Each is registered in `FEATURES`, so the
look-ahead guard ([page 5](05-backtesting-and-signals.md)) checks it automatically.

| Feature | Definition |
|---|---|
| `ret_5`, `ret_21`, `ret_63`, `ret_126`, `ret_252` | `P_t / P_{t−n} − 1` |
| `dist_sma50`, `dist_sma200` | `P_t / SMA_n − 1` |
| `rsi_14` | Wilder RSI(14) |
| `vol_21` | 21-day realized volatility, annualized |
| `ewma_vol_ratio` | EWMA volatility (λ = 0.94) ÷ 252-day realized volatility |
| `regime` | Causal volatility regime as 0 / 1 / 2 ([page 3](03-risk-and-distribution.md)) |
| `drawdown_252` | `P_t / max(P over the last 252 days) − 1` |
| `beta_252`, `rel_strength_63` | 252-day rolling beta and 63-day return relative to the benchmark (only when a benchmark is available) |

The first 252 rows are dropped as warm-up. An implied-minus-realized volatility spread is not included, because
implied volatility data isn't available.

**Label:** `1` if `P_{t+H}/P_t − 1` exceeds the risk-free rate compounded over the same `H` days, else `0`. The last
`H` labels are unknown (NaN). Labels look forward by construction and are never features; the guard test asserts that
they *are* flagged as forward-looking.

## Training (`stockapp/model/probability.py`, `stockapp/model/cv.py`)

**Walk-forward folds:** the same calendar folds as the backtests (at least 3 years of training, 1-year test steps,
expanding), built on the post-warm-up index. At each fold's last training close, only samples whose label window has
already ended are used (sample index ≤ train end − H). The model is then frozen for the test year.

**Within each training window:**

1. The last 20% is held out for calibration. The fit part ends `H` samples before it, so no fit label overlaps the
   calibration period.
2. Hyperparameters are chosen by **purged, embargoed K-fold CV** (K = 3, López de Prado ch. 7) on the fit part. Each
   test fold drops from training every sample whose label window overlaps it, plus an embargo of `H` samples after it,
   because overlapping forward labels otherwise leak information. Selection is by mean log loss over a fixed 4-point
   grid: `num_leaves ∈ {7, 15} × n_estimators ∈ {100, 200}`. The fixed settings are learning rate 0.05,
   `min_child_samples = 100`, `colsample_bytree = 0.8` and seed 0. Folds whose training labels are single-class are
   skipped.
3. [LightGBM](https://lightgbm.readthedocs.io/en/latest/) is refitted on the fit part with the chosen grid point.
4. [Isotonic regression](https://scikit-learn.org/stable/modules/isotonic.html) on the held-out tail maps raw scores to
   calibrated probabilities (clipped to [0, 1]).

**Climatology** for each fold is the training window's base rate of positive labels: the no-skill forecast.

## Evaluation and the skill test

The test-year predictions are stitched into one out-of-sample series. Over the dates whose outcome is known, the app
computes:

- Brier score vs climatology's Brier score
- log loss vs climatology's log loss (probabilities clipped to [1e-6, 1 − 1e-6])
- AUC
- the number of predictions and the out-of-sample base rate
- a 10-bin reliability table and diagram

**`has_skill` ⇔ out-of-sample Brier < climatology Brier.** This is a description of the model, shown on the Model
tab. Whether the *advice* uses p is decided by a stricter gate, `skill_established`: the advice rule's causal skill
gate evaluated at the latest date, which also requires at least 252 scored predictions
([page 7](07-advice-rules.md)). The reported trial count is 4 (the grid size).

**Latest prediction:** a model fitted on every sample with a known label scores the latest bar. Feature contributions
are LightGBM's `pred_contrib` values: contributions to the *raw* (pre-calibration) log-odds, without the bias term. The
page shows the top five.

Overlapping `H`-day labels mean consecutive predictions are highly dependent. The effective sample size is far smaller
than the prediction count, and a small Brier improvement is weak evidence. That is part of why the advice rule
([page 7](07-advice-rules.md)) gates the probability in its own backtest more strictly.

## References

- M. López de Prado, [*Advances in Financial Machine Learning* (Wiley, 2018)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086),
  ch. 7 (purged and embargoed cross-validation)
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html) and
  [isotonic regression](https://scikit-learn.org/stable/modules/isotonic.html)
- Wikipedia: [Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting),
  [Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)),
  [Brier score](https://en.wikipedia.org/wiki/Brier_score),
  [ROC / AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic),
  [Isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression)

[← Backtesting and signals](05-backtesting-and-signals.md) · Next: [Advice rules →](07-advice-rules.md)
