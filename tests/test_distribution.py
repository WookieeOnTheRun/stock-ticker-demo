import math

import numpy as np
import pandas as pd
import pytest
from fixtures import make_prices

from stockapp import distribution


def _returns(seed, dist="normal", n=2500):
    return make_prices(n + 1, mu=0.0003, sigma=0.01, seed=seed, dist=dist).pct_change().dropna()


def test_normal_samples_look_normal():
    p_values = []
    for seed in range(20):
        r = _returns(seed)
        assert abs(distribution.excess_kurtosis(r)) < 0.5
        p_values.append(distribution.jarque_bera(r)[1])
    assert np.mean(np.array(p_values) > 0.01) >= 0.9  # mostly not rejected


def test_t3_sample_is_fat_tailed_and_rejected():
    r = _returns(seed=1, dist="t")
    assert distribution.excess_kurtosis(r) > 3
    assert distribution.jarque_bera(r)[1] < 0.01


def test_jarque_bera_p_value_is_chi_square_two_df():
    r = _returns(seed=2, dist="t")
    jb, p = distribution.jarque_bera(r)
    assert p == pytest.approx(math.exp(-jb / 2))


def test_qq_normal_uses_plotting_positions():
    qq = distribution.qq_normal(pd.Series([0.01, -0.02, 0.03, 0.0]))
    assert qq["theoretical"].iloc[0] == pytest.approx(-1.1503493803760079)  # Phi^-1((1 - 0.5) / 4)
    assert qq["sample"].is_monotonic_increasing


def test_student_t_fit_recovers_heavy_tails():
    r = _returns(seed=3, dist="t", n=5000)
    assert distribution.fit_student_t(r).df < 6
    assert distribution.fit_student_t(_returns(seed=3, n=5000)).df > 20


def test_acf_and_ljung_box_detect_autocorrelation():
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(2000)
    ar = np.empty(2000)
    ar[0] = noise[0]
    for t in range(1, 2000):
        ar[t] = 0.5 * ar[t - 1] + noise[t]
    assert distribution.acf(pd.Series(ar))[0] == pytest.approx(0.5, abs=0.05)
    assert distribution.ljung_box(pd.Series(ar))[1] < 0.01
    assert distribution.ljung_box(pd.Series(noise))[1] > 0.01
    assert distribution.acf_band(400) == pytest.approx(0.098)


def test_hurst_near_half_for_random_walk_returns():
    assert 0.4 < distribution.hurst_rs(_returns(seed=6, n=4000)) < 0.65


def test_verdict_flags_fat_tails_only_when_present():
    assert not distribution.gaussian_kelly_verdict(_returns(seed=8)).fat_tails
    verdict = distribution.gaussian_kelly_verdict(_returns(seed=8, dist="t"))
    assert verdict.fat_tails and verdict.non_normal
    assert verdict.message.startswith("Not entirely")
