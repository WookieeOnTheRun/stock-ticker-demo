import io
import logging
import re
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from stockapp.providers.errors import DataError, ProviderUnavailable, TickerNotFound
from stockapp.providers.yfinance_provider import with_retries
from stockapp.store import FACTOR_COLUMNS, Store

logger = logging.getLogger(__name__)

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_FILE = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOMENTUM_FILE = "F-F_Momentum_Factor_daily_CSV.zip"
TIMEOUT_S = 30
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_ZIP_ENTRIES = 5
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
REFRESH_EVERY = timedelta(days=7)
NEWEY_WEST_LAG = 5

# Column headers as published (verified on the real files, October 2026; data through 2026-08-31).
FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
MOMENTUM_COLUMNS = ["Mom"]
COLUMN_NAMES = {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma", "RF": "rf", "Mom": "mom"}
REGRESSORS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]


class FactorFormatError(DataError):
    """A Ken French file is not in the layout this parser was written against."""


def download(url: str, timeout: float = TIMEOUT_S, max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    """GET over HTTPS with a timeout and a size cap; retries transient failures (see with_retries)."""
    if not url.startswith("https://"):
        raise ValueError("Factor files must be downloaded over HTTPS")

    def attempt() -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "stockapp-research"})  # noqa: S310 - https only
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https only
                data: bytes = response.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise TickerNotFound(f"Factor file not found: {url.rsplit('/', 1)[-1]}") from exc
            raise ProviderUnavailable(f"Factor download failed (HTTP {exc.code})") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderUnavailable(f"Could not reach the factor data library ({type(exc).__name__})") from exc
        if len(data) > max_bytes:
            raise FactorFormatError(f"Factor file exceeds the {max_bytes // (1024 * 1024)} MB limit")
        return data

    return with_retries(attempt)


def read_single_csv(archive: bytes) -> str:
    """
    Unzip in memory and return the one CSV's text. Rejects archives with more than MAX_ZIP_ENTRIES entries, any
    entry name containing ".." or starting with "/", oversized entries, or anything other than exactly one CSV.
    Nothing is written to disk or executed.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise FactorFormatError("Factor download is not a zip archive") from exc
    entries = zf.infolist()
    if len(entries) > MAX_ZIP_ENTRIES:
        raise FactorFormatError(f"Zip has {len(entries)} entries (max {MAX_ZIP_ENTRIES})")
    for info in entries:
        if ".." in info.filename or info.filename.startswith(("/", "\\")):
            raise FactorFormatError(f"Unsafe path in zip: {info.filename!r}")
        if info.file_size > MAX_UNCOMPRESSED_BYTES:
            raise FactorFormatError(f"Zip entry too large: {info.filename!r}")
    csvs = [info for info in entries if info.filename.lower().endswith(".csv")]
    if len(csvs) != 1:
        raise FactorFormatError(f"Expected exactly one CSV in the zip, found {len(csvs)}")
    return zf.read(csvs[0]).decode("latin-1")


DATA_ROW = re.compile(r"^\d{8},")


def parse_french_csv(text: str, expected_columns: list[str]) -> pd.DataFrame:
    """
    Parse a Ken French daily CSV with the layout verified in October 2026:
      free-text preamble (variable length) -> one header row starting with "," -> contiguous "YYYYMMDD,v1,v2,..."
      rows -> a blank line and a "Copyright ..." footer.
    Values are percent and are returned as decimals (divided by 100), indexed by a DatetimeIndex named "Date".
    Raises FactorFormatError if anything deviates from that layout.
    """
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith(",")]
    if len(headers) != 1:
        raise FactorFormatError(f"Expected one header row starting with ',', found {len(headers)}")
    header = headers[0]
    columns = [c.strip() for c in lines[header].split(",")[1:]]
    if columns != expected_columns:
        raise FactorFormatError(f"Unexpected columns {columns}, expected {expected_columns}")
    end = header + 1
    while end < len(lines) and DATA_ROW.match(lines[end]):
        end += 1
    if end == header + 1:
        raise FactorFormatError("No data rows after the header")
    trailing = [line.strip() for line in lines[end:] if line.strip()]
    if any(not line.startswith("Copyright") for line in trailing):
        raise FactorFormatError("Unexpected content after the daily data block (a second table?)")
    frame = pd.read_csv(
        io.StringIO("\n".join(lines[header + 1 : end])),
        header=None,
        names=["date", *expected_columns],
        dtype={"date": str},
        skipinitialspace=True,
    )
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), format="%Y%m%d"), name="Date")
    frame = frame.astype("float64")
    if "Mkt-RF" in frame and not (frame["Mkt-RF"].abs().quantile(0.99) < 20):
        raise FactorFormatError("Mkt-RF magnitudes are not in percent units")
    return (frame / 100).rename(columns=COLUMN_NAMES)


def fetch_factors() -> pd.DataFrame:
    """Download and parse the daily FF5 (2x3) and Momentum factors; inner-joined on date, decimal returns."""
    ff5 = parse_french_csv(read_single_csv(download(BASE_URL + FF5_FILE)), FF5_COLUMNS)
    mom = parse_french_csv(read_single_csv(download(BASE_URL + MOMENTUM_FILE)), MOMENTUM_COLUMNS)
    return ff5.join(mom, how="inner")[FACTOR_COLUMNS]


def load_factors(store: Store | None = None, now: datetime | None = None) -> pd.DataFrame:
    """
    Factors from the SQLite cache, refreshed from the Ken French library at most once every REFRESH_EVERY.
    If a refresh fails and a cache exists, the cache is returned.
    """
    store = store if store is not None else Store()
    now = now if now is not None else datetime.now(UTC)
    cached, fetched = store.get_factors()
    if fetched is not None and now - fetched < REFRESH_EVERY and not cached.empty:
        return cached
    try:
        factors = fetch_factors()
    except DataError as exc:
        if cached.empty:
            raise
        logger.warning("Serving cached factors: %s", exc)
        return cached
    store.replace_factors(factors, now)
    return factors


def newey_west_cov(x: np.ndarray, resid: np.ndarray, lag: int = NEWEY_WEST_LAG) -> np.ndarray:
    """
    Newey-West HAC covariance of OLS coefficients with Bartlett weights w_l = 1 - l / (lag + 1):
      S = sum_t u_t^2 x_t x_t' + sum_{l=1..lag} w_l sum_{t>l} u_t u_{t-l} (x_t x_{t-l}' + x_{t-l} x_t')
      V = (X'X)^-1 S (X'X)^-1
    No small-sample degrees-of-freedom correction is applied.
    """
    scores = x * resid[:, None]
    s = scores.T @ scores
    for lag_l in range(1, lag + 1):
        weight = 1 - lag_l / (lag + 1)
        gamma = scores[lag_l:].T @ scores[:-lag_l]
        s += weight * (gamma + gamma.T)
    xtx_inv = np.linalg.inv(x.T @ x)
    covariance: np.ndarray = xtx_inv @ s @ xtx_inv
    return covariance


@dataclass(frozen=True)
class FactorFit:
    coefs: pd.Series
    """Intercept ("alpha", daily) and factor loadings."""
    std_errors: pd.Series
    t_stats: pd.Series
    r2: float
    n_obs: int


def stock_excess_returns(r: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """Stock return minus the factor file's own RF, on dates present in both (keeps the regression consistent)."""
    joined = pd.concat([r.rename("r"), factors["rf"]], axis=1, join="inner").dropna()
    return (joined["r"] - joined["rf"]).rename("excess")


def factor_regression(
    excess_r: pd.Series, factors: pd.DataFrame, regressors: list[str] = REGRESSORS, lag: int = NEWEY_WEST_LAG
) -> FactorFit:
    """
    OLS of daily excess returns on the factors (numpy.linalg.lstsq) over common dates, with Newey-West (HAC, lag 5)
    standard errors and t-statistics.
    """
    joined = pd.concat([excess_r.rename("y"), factors[regressors]], axis=1, join="inner").dropna()
    y = joined["y"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(joined)), joined[regressors].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    se = np.sqrt(np.diag(newey_west_cov(x, resid, lag)))
    names = ["alpha", *regressors]
    total = float(((y - y.mean()) ** 2).sum())
    return FactorFit(
        coefs=pd.Series(beta, index=names),
        std_errors=pd.Series(se, index=names),
        t_stats=pd.Series(beta / se, index=names),
        r2=1 - float((resid**2).sum()) / total if total > 0 else 0.0,
        n_obs=len(y),
    )


FACTOR_LABELS = {
    "mkt_rf": "market",
    "smb": "size (small caps)",
    "hml": "value",
    "rmw": "profitability",
    "cma": "conservative investment",
    "mom": "momentum",
}


def describe_fit(fit: FactorFit) -> str:
    """One-sentence plain-language read of a factor fit, naming loadings with |t| > 2."""
    beta = fit.coefs["mkt_rf"]
    if fit.r2 >= 0.6:
        lead = "Most of this stock's movement is market beta"
    elif fit.r2 >= 0.3:
        lead = "Market and factors explain part of this stock's movement"
    else:
        lead = "Market and factors explain little of this stock's movement"
    tilts = []
    for name in REGRESSORS[1:]:
        if abs(fit.t_stats[name]) > 2:
            direction = "tilt" if fit.coefs[name] > 0 else "negative tilt"
            tilts.append(f"{FACTOR_LABELS[name]} {direction}")
    tail = f"; significant {', '.join(tilts)}." if tilts else "; no significant factor tilts beyond the market."
    return f"{lead} (beta {beta:.2f}, R² {fit.r2:.2f}){tail}"
