import time
from dataclasses import dataclass

import pandas as pd

from stockapp.bootstrap import SEED, block_bootstrap_indices
from stockapp.estimation import estimate
from stockapp.optimize import ConstrainedKelly

N_RESAMPLES = 200
MIN_RESAMPLES = 30
PILOT_RESAMPLES = 10
TIME_BUDGET_S = 1.2  # share of the 2 s portfolio budget given to the bootstrap
BLOCK_DAILY = 20  # about a month of trading days
BLOCK_WEEKLY = 4  # about a month of weeks


@dataclass(frozen=True)
class WeightBands:
    p5: pd.Series
    p50: pd.Series
    p95: pd.Series
    resamples: int
    requested: int
    fallbacks: int
    """Resamples where the optimizer failed and equal weight was used."""

    @property
    def reduced(self) -> bool:
        return self.resamples < self.requested


def bootstrap_weights(
    excess: pd.DataFrame,
    prior_sd_annual: float,
    periods_per_year: float,
    solver: ConstrainedKelly,
    c: float,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    time_budget_s: float = TIME_BUDGET_S,
) -> WeightBands:
    """
    Block-bootstrap the rows of the excess-return matrix (the same rows for every asset, so cross-correlation is
    preserved), re-estimate mu and Sigma and re-solve the constrained Kelly problem on each resample, and report each
    weight's 5th / 50th / 95th percentile. Blocks are 20 periods for daily data and 4 for weekly. Resamples 2 to
    PILOT_RESAMPLES + 1 are timed (the first one includes one-off solver set-up); if the full count would exceed
    time_budget_s, fewer resamples are run (at least MIN_RESAMPLES) and `reduced` is True.
    """
    values = excess.dropna()
    block = BLOCK_WEEKLY if periods_per_year < 100 else BLOCK_DAILY
    idx = block_bootstrap_indices(len(values), n_resamples=n_resamples, block=block, seed=seed)
    matrix = values.to_numpy(dtype=float)
    weights, fallbacks = [], 0
    target = n_resamples
    start = time.perf_counter()
    for k in range(n_resamples):
        if k == 1:
            start = time.perf_counter()
        if k == PILOT_RESAMPLES + 1:
            per = (time.perf_counter() - start) / PILOT_RESAMPLES
            target = max(MIN_RESAMPLES, min(n_resamples, int(time_budget_s / per))) if per > 0 else n_resamples
        if k >= target:
            break
        sample = pd.DataFrame(matrix[idx[k]], columns=values.columns)
        est = estimate(sample, prior_sd_annual, periods_per_year, diagnostics=False)
        allocation = solver.solve(est.mu, est.sigma, c)
        fallbacks += allocation.fallback
        weights.append(allocation.weights)
    frame = pd.DataFrame(weights, columns=values.columns)
    return WeightBands(
        p5=frame.quantile(0.05),
        p50=frame.quantile(0.50),
        p95=frame.quantile(0.95),
        resamples=len(frame),
        requested=n_resamples,
        fallbacks=fallbacks,
    )
