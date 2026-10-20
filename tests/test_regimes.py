import numpy as np
import pandas as pd
import pytest

from stockapp import regimes


def _regime_switching_returns(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < 0.99 else 1 - states[t - 1]
    r = rng.standard_normal(n) * np.where(states == 1, 0.025, 0.008) + 0.0003
    return pd.Series(r, index=pd.bdate_range("2015-01-01", periods=n)), states


def test_labels_do_not_change_when_future_data_is_appended():
    r, _ = _regime_switching_returns()
    full = regimes.vol_regimes(r)
    for cut in (400, 900, 1500):
        pd.testing.assert_series_equal(regimes.vol_regimes(r.iloc[:cut]), full.iloc[:cut])


def test_labels_track_volatility_and_need_history():
    r, _ = _regime_switching_returns()
    labels = regimes.vol_regimes(r)
    assert labels.iloc[: regimes.VOL_WINDOW + regimes.MIN_HISTORY - 2].isna().all()
    vol = r.rolling(regimes.VOL_WINDOW).std()
    assert vol[labels == "high"].mean() > vol[labels == "normal"].mean() > vol[labels == "low"].mean()


def test_kelly_by_regime_counts_days_and_uses_excess_returns():
    r, _ = _regime_switching_returns()
    rf = pd.Series(0.0001, index=r.index)
    labels = regimes.vol_regimes(r)
    table = regimes.kelly_by_regime(r, rf, labels)
    assert list(table.index) == ["low", "normal", "high"]
    assert table["days"].sum() == labels.notna().sum()
    low_days = labels.index[labels == "low"]
    e = r[low_days] - 0.0001
    assert table.loc["low", "gaussian_kelly"] == pytest.approx(e.mean() / e.var())


# hmmlearn 0.3.3 sets ndarray.shape, deprecated in NumPy 2.5 (upstream issue, harmless for now).
@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_hmm_recovers_two_regimes_with_ordered_names():
    r, states = _regime_switching_returns(n=2500, seed=1)
    labels = regimes.hmm_regimes(r, n_states=2)
    assert set(labels) == {"low", "high"}
    accuracy = np.mean((labels.to_numpy() == "high") == (states == 1))
    assert accuracy > 0.9
