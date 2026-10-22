"""Split-sample evaluation of fixed portfolio weights."""

import numpy as np
import pandas as pd


def split_halves(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First and second half of the rows (in-sample / out-of-sample split)."""
    half = len(returns) // 2
    return returns.iloc[:half], returns.iloc[half:]


def fixed_weight_growth(returns: pd.DataFrame, rf: pd.Series, weights: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Growth of $1 for each fixed weight vector, rebalanced every period to the same weights, with the remainder
    (1 - sum of weights, negative if leveraged) earning the risk-free rate. Columns are the weighting names.
    """
    curves = {}
    rf_aligned = rf.reindex(returns.index).fillna(0.0)
    for name, w in weights.items():
        aligned = w.reindex(returns.columns).fillna(0.0)
        period = returns.to_numpy(dtype=float) @ aligned.to_numpy(dtype=float) + (1 - aligned.sum()) * rf_aligned
        curves[name] = np.cumprod(1 + period)
    growth = pd.DataFrame(curves, index=returns.index)
    start = pd.DataFrame({name: [1.0] for name in weights}, index=[returns.index[0] - pd.Timedelta(days=1)])
    return pd.concat([start, growth])
