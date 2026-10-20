import io
import zipfile
from datetime import UTC, datetime, timedelta
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from stockapp import factors
from stockapp.factors import FactorFormatError
from stockapp.providers.errors import ProviderUnavailable
from stockapp.store import Store

# Mimics the verified layout of F-F_Research_Data_5_Factors_2x3_daily.csv (October 2026): a free-text preamble,
# one header row starting with ",", space-padded YYYYMMDD rows in percent, a blank line and a copyright footer.
FF5_TEXT = "\r\n".join(
    [
        "This file was created by using the 202608 CRSP database.",
        "The Tbill return is the simple daily rate that, over the number of trading days",
        "compounds to 1-month TBill rate.",
        "",
        ",Mkt-RF,SMB,HML,RMW,CMA,RF",
        "20260826,   -0.06,   -0.13,    0.20,    0.26,    0.46,    0.01",
        "20260827,    0.72,   -0.17,   -1.14,   -0.78,   -0.89,    0.01",
        "20260828,   -0.34,   -0.37,    0.28,    1.66,    0.49,    0.01",
        "",
        "Copyright 2026 Eugene F. Fama and Kenneth R. French",
    ]
)
MOM_TEXT = "\r\n".join(
    [
        "This file was created by using the 202608 CRSP database.  It",
        "contains a momentum factor.",
        "",
        ",Mom",
        "20260827,  -0.23",
        "20260828,  -1.47",
        "",
        "Copyright 2026 Eugene F. Fama and Kenneth R. French",
    ]
)


def _zip(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, text in entries.items():
            zf.writestr(name, text)
    return buffer.getvalue()


def test_parser_reads_layout_and_converts_percent():
    frame = factors.parse_french_csv(FF5_TEXT, factors.FF5_COLUMNS)
    assert list(frame.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert frame.index.name == "Date"
    assert frame.index[0] == pd.Timestamp("2026-08-26")
    assert frame.loc["2026-08-27", "mkt_rf"] == pytest.approx(0.0072)
    assert len(frame) == 3


def test_parser_fails_loudly_on_layout_changes():
    with pytest.raises(FactorFormatError, match="columns"):
        factors.parse_french_csv(FF5_TEXT.replace("RMW", "XYZ"), factors.FF5_COLUMNS)
    with pytest.raises(FactorFormatError, match="after the daily data"):
        factors.parse_french_csv(
            FF5_TEXT.replace("Copyright", " Annual Factors: January-December"), factors.FF5_COLUMNS
        )
    with pytest.raises(FactorFormatError, match="header"):
        factors.parse_french_csv(FF5_TEXT.replace(",Mkt-RF", "Mkt-RF"), factors.FF5_COLUMNS)
    with pytest.raises(FactorFormatError, match="percent"):
        factors.parse_french_csv(FF5_TEXT.replace("0.72", "72.00"), factors.FF5_COLUMNS)


def test_zip_safety_checks():
    assert factors.read_single_csv(_zip({"F-F_Momentum_Factor_daily.csv": MOM_TEXT})) == MOM_TEXT
    with pytest.raises(FactorFormatError, match="entries"):
        factors.read_single_csv(_zip({f"f{i}.csv": "x" for i in range(6)}))
    with pytest.raises(FactorFormatError, match="Unsafe path"):
        factors.read_single_csv(_zip({"../evil.csv": "x"}))
    with pytest.raises(FactorFormatError, match="exactly one CSV"):
        factors.read_single_csv(_zip({"a.csv": "x", "b.csv": "y"}))
    with pytest.raises(FactorFormatError, match="not a zip"):
        factors.read_single_csv(b"not a zip")


def test_download_requires_https():
    with pytest.raises(ValueError, match="HTTPS"):
        factors.download("http://example.com/file.zip")


def test_newey_west_matches_hand_computed_reference():
    # y on [1, x], lag 1 (Bartlett weight 1/2). Worked by hand with exact fractions:
    #   OLS a = -1/30, b = 69/70; residuals 1/21, 59/105, -97/105, 62/105, -94/105, 13/21
    #   S = sum u_t^2 x_t x_t' + 1/2 * sum u_t u_{t-1} (x_t x_{t-1}' + x_{t-1} x_t');  V = (X'X)^-1 S (X'X)^-1
    #   V[0, 0] = 36476/354375, V[1, 1] = 5452/643125
    x = np.column_stack([np.ones(6), np.arange(1, 7, dtype=float)])
    y = np.array([1.0, 2.5, 2.0, 4.5, 4.0, 6.5])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    assert beta == pytest.approx([-1 / 30, 69 / 70])
    cov = factors.newey_west_cov(x, y - x @ beta, lag=1)
    assert cov[0, 0] == pytest.approx(float(Fraction(36476, 354375)), rel=1e-12)
    assert cov[1, 1] == pytest.approx(float(Fraction(5452, 643125)), rel=1e-12)


def test_factor_regression_recovers_loadings():
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2016-01-01", periods=2000)
    f = pd.DataFrame(rng.normal(0, 0.01, (2000, 6)), index=index, columns=factors.REGRESSORS)
    f["rf"] = 0.0001
    true = np.array([1.2, 0.3, -0.4, 0.0, 0.0, 0.1])
    r = pd.Series(f[factors.REGRESSORS].to_numpy() @ true + 0.0001 + rng.normal(0, 0.005, 2000), index=index)

    fit = factors.factor_regression(factors.stock_excess_returns(r, f), f)

    assert fit.coefs[factors.REGRESSORS].to_numpy() == pytest.approx(true, abs=0.05)
    assert abs(fit.t_stats["mkt_rf"]) > 10
    assert abs(fit.t_stats["rmw"]) < 3
    assert fit.n_obs == 2000
    assert "market beta" in factors.describe_fit(fit)


def test_load_factors_refreshes_at_most_weekly(monkeypatch):
    calls = []
    sample = factors.parse_french_csv(FF5_TEXT, factors.FF5_COLUMNS).join(
        factors.parse_french_csv(MOM_TEXT, factors.MOMENTUM_COLUMNS), how="inner"
    )

    def fake_fetch():
        calls.append(1)
        return sample

    monkeypatch.setattr(factors, "fetch_factors", fake_fetch)
    now = datetime(2026, 10, 20, tzinfo=UTC)
    store = Store()
    first = factors.load_factors(store, now)
    factors.load_factors(store, now + timedelta(days=6))
    assert len(calls) == 1
    factors.load_factors(store, now + timedelta(days=8))
    assert len(calls) == 2
    assert first.index.tolist() == [pd.Timestamp("2026-08-27"), pd.Timestamp("2026-08-28")]

    def failing():
        raise ProviderUnavailable("down")

    monkeypatch.setattr(factors, "fetch_factors", failing)
    served = factors.load_factors(store, now + timedelta(days=20))  # stale, refresh fails: cache served
    pd.testing.assert_frame_equal(served, sample, check_freq=False)
