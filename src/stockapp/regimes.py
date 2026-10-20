import numpy as np
import pandas as pd

from stockapp.kelly import excess_returns, fractional_kelly, shrunk_kelly

VOL_WINDOW = 21
MIN_HISTORY = 252  # rolling-vol observations required before the expanding percentiles are trusted
LOW_Q, HIGH_Q = 0.33, 0.67
REGIME_ORDER = ["low", "normal", "high"]


def vol_regimes(r: pd.Series, window: int = VOL_WINDOW, min_history: int = MIN_HISTORY) -> pd.Series:
    """
    Causal volatility regimes. Each day's 21-day rolling volatility is compared with the 33rd and 67th percentiles
    of all rolling-vol values up to and including that day (expanding window), giving "low", "normal" or "high".
    Only past and current data enter each label, so appending future data never changes an existing label.
    Days before `window + min_history - 1` returns are unlabelled (NaN). A label at t includes day t's return, so
    a rule that acts on day t should use the label from t - 1.
    """
    vol = r.rolling(window).std()
    expanding = vol.expanding(min_periods=min_history)
    low, high = expanding.quantile(LOW_Q), expanding.quantile(HIGH_Q)
    labels = pd.Series(np.where(vol <= low, "low", np.where(vol >= high, "high", "normal")), index=r.index)
    return labels.where(low.notna() & vol.notna()).rename("regime")


def hmm_regimes(r: pd.Series, n_states: int = 2, seed: int = 0) -> pd.Series:
    """
    Gaussian HMM regimes (hmmlearn) - HISTORICAL VIEW ONLY. The Viterbi path is decoded from the whole sample,
    so every label depends on future data; never feed these labels into signals or sizing. hmmlearn 0.3.3 exposes
    no forward-only (filtered) probabilities, so no causal variant is provided. States are ordered by variance and
    named low/high (2 states) or low/normal/high (3 states). Returns are scaled to percent before fitting.
    """
    from hmmlearn.hmm import GaussianHMM  # imported lazily: optional, slower to import

    clean = r.dropna()
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=seed)
    observations = clean.to_numpy(dtype=float).reshape(-1, 1) * 100
    model.fit(observations)
    states = model.predict(observations)
    names = ["low", "high"] if n_states == 2 else REGIME_ORDER
    order = np.argsort(np.asarray(model.covars_).reshape(n_states, -1)[:, 0])
    rank = np.empty(n_states, dtype=int)
    rank[order] = np.arange(n_states)
    return pd.Series([names[rank[s]] for s in states], index=clean.index, name="regime")


def kelly_by_regime(r: pd.Series, rf: pd.Series, labels: pd.Series, prior_sd_annual: float = 0.10) -> pd.DataFrame:
    """
    Gaussian Kelly (mean / variance of excess returns) and shrunk Kelly per regime, with the number of days in each.
    Days without a label are excluded. Rows follow low / normal / high order.
    """
    rows = []
    known = labels.dropna()
    for regime in [name for name in REGIME_ORDER if name in set(known)]:
        days = known.index[known == regime]
        r_regime = r.reindex(days).dropna()
        rows.append(
            {
                "regime": regime,
                "days": len(r_regime),
                "gaussian_kelly": fractional_kelly(r_regime, rf=rf).raw_kelly,
                "shrunk_kelly": shrunk_kelly(excess_returns(r_regime, rf), prior_sd_annual),
            }
        )
    return pd.DataFrame(rows, columns=["regime", "days", "gaussian_kelly", "shrunk_kelly"]).set_index("regime")
