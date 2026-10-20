from typing import Literal

import numpy as np
import pandas as pd


def make_prices(
    n: int,
    mu: float,
    sigma: float,
    seed: int,
    dist: Literal["normal", "t"] = "normal",
    df: float = 3,
) -> pd.Series:
    """
    Deterministic synthetic daily closes starting at 100.
    Daily log returns have mean mu and standard deviation sigma; dist="t" draws Student-t shocks
    with df degrees of freedom (df > 2), rescaled to unit variance, for fat tails.
    Returns a float Series named "Close" with a business-day DatetimeIndex named "Date".
    """
    rng = np.random.default_rng(seed)
    if dist == "normal":
        shocks = rng.standard_normal(n)
    elif dist == "t":
        if df <= 2:
            raise ValueError("df must be > 2 for the t distribution to have finite variance")
        shocks = rng.standard_t(df, size=n) * np.sqrt((df - 2) / df)
    else:
        raise ValueError(f"Unknown dist: {dist!r}")
    log_returns = mu + sigma * shocks
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    index = pd.bdate_range("2020-01-01", periods=n, name="Date")
    return pd.Series(prices, index=index, name="Close")
