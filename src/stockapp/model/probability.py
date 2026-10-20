import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from stockapp.backtest.walkforward import make_folds
from stockapp.model.cv import purged_kfold
from stockapp.model.features import WARMUP, build_features, forward_excess_label

logger = logging.getLogger(__name__)

HYPER_GRID: tuple[dict[str, int], ...] = tuple(
    {"num_leaves": leaves, "n_estimators": trees} for leaves in (7, 15) for trees in (100, 200)
)
"""Small fixed grid (4 combinations, counted as trials); learning rate and leaf size are fixed below."""
LEARNING_RATE = 0.05
MIN_CHILD_SAMPLES = 100
COLSAMPLE_BYTREE = 0.8
CV_FOLDS = 3
CALIBRATION_SHARE = 0.2
SEED = 0
RELIABILITY_BINS = 10
EPS = 1e-6


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Brier score: mean squared difference between the predicted probability and the 0/1 outcome."""
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the 0/1 outcomes, with probabilities clipped to [1e-6, 1 - 1e-6]."""
    q = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def _classifier(params: dict[str, int]) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        num_leaves=params["num_leaves"],
        n_estimators=params["n_estimators"],
        learning_rate=LEARNING_RATE,
        min_child_samples=MIN_CHILD_SAMPLES,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=SEED,
        verbose=-1,
    )


def _proba(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    """P(label = 1) as a dense array (LightGBM's predict_proba is typed as possibly sparse)."""
    return np.asarray(model.predict_proba(x))[:, 1]


@dataclass
class WindowModel:
    """A LightGBM classifier fitted on one training window plus an isotonic calibrator fitted on its held-out tail."""

    model: lgb.LGBMClassifier
    calibrator: IsotonicRegression
    params: dict[str, int]
    base_rate: float
    """Share of positive labels in the training window (the climatology forecast)."""
    features: list[str] = field(default_factory=list)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raw = _proba(self.model, x[self.features])
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    def contributions(self, x: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contributions to the raw log-odds (LightGBM pred_contrib), without the bias column."""
        contrib = np.asarray(self.model.predict(x[self.features], pred_contrib=True))
        return pd.DataFrame(contrib[:, :-1], index=x.index, columns=self.features)


def fit_window(x: pd.DataFrame, y: pd.Series, horizon: int) -> WindowModel:
    """
    Fit on one training window of samples whose labels are known. The last CALIBRATION_SHARE is held out for
    isotonic calibration; the fit part ends `horizon` samples before it so no fit label overlaps the calibration
    period. Hyperparameters come from purged K-fold CV (mean log loss) on the fit part.
    """
    n = len(x)
    split = int(n * (1 - CALIBRATION_SHARE))
    x_fit, y_fit = x.iloc[: max(split - horizon, 0)], y.iloc[: max(split - horizon, 0)]
    x_cal, y_cal = x.iloc[split:], y.iloc[split:]
    best, best_loss = HYPER_GRID[0], np.inf
    folds = purged_kfold(len(x_fit), horizon, CV_FOLDS)
    for params in HYPER_GRID:
        losses = []
        for train, test in folds:
            if y_fit.iloc[train].nunique() < 2:
                continue
            clf = _classifier(params).fit(x_fit.iloc[train], y_fit.iloc[train])
            losses.append(log_loss(_proba(clf, x_fit.iloc[test]), y_fit.iloc[test].to_numpy()))
        if losses and float(np.mean(losses)) < best_loss:
            best, best_loss = params, float(np.mean(losses))
    model = _classifier(best).fit(x_fit, y_fit)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(_proba(model, x_cal), y_cal.to_numpy())
    return WindowModel(model, calibrator, dict(best), float(y.mean()), list(x.columns))


@dataclass(frozen=True)
class ModelReport:
    horizon: int
    probabilities: pd.Series
    """Stitched out-of-sample calibrated P(excess return > 0 over the horizon), one per decision date."""
    climatology: pd.Series
    """The training base rate in force on each decision date (the no-skill forecast)."""
    labels: pd.Series
    """Realized labels for the decision dates (NaN where not yet known)."""
    metrics: dict[str, float]
    reliability: pd.DataFrame
    has_skill: bool
    """True only if the out-of-sample Brier score beats climatology."""
    latest_probability: float
    latest_contributions: pd.Series
    trials: int
    fold_params: list[dict[str, int]]


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = RELIABILITY_BINS) -> pd.DataFrame:
    """Calibration curve data: per probability bin, the mean prediction, the observed frequency and the count."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        rows.append(
            {
                "bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}",
                "mean_predicted": float(p[mask].mean()) if mask.any() else np.nan,
                "observed": float(y[mask].mean()) if mask.any() else np.nan,
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_model(
    prices: pd.Series, rf_daily: pd.Series, horizon: int = 21, benchmark: pd.Series | None = None
) -> ModelReport:
    """
    Walk-forward probability model on the A4 folds (3-year minimum training, 1-year test steps). At each fold's
    last training close only samples whose labels are already known (label end <= that close) are used; the model
    is then frozen for the test year. Out-of-sample metrics: Brier vs climatology, log loss, AUC, reliability.
    The latest-bar probability and feature contributions come from a model fitted on every currently known label.
    """
    features = build_features(prices, benchmark).iloc[WARMUP:]
    labels = forward_excess_label(prices, rf_daily, horizon).reindex(features.index)
    positions = {d: i for i, d in enumerate(features.index)}
    folds = make_folds(pd.DatetimeIndex(features.index))
    if not folds:
        raise ValueError("The model needs more than 3 years of history after a one-year feature warm-up")
    probs, clim, fold_params = [], [], []
    for k, (train_start, train_end, test_end) in enumerate(folds):
        end = positions[train_end] - horizon  # last sample whose label is known at the train_end close
        start = positions[train_start]
        x_train, y_train = features.iloc[start : end + 1], labels.iloc[start : end + 1]
        window = fit_window(x_train, y_train, horizon)
        fold_params.append(window.params)
        decisions = features.loc[train_end:test_end]
        if k < len(folds) - 1:
            decisions = decisions.iloc[:-1]
        probs.append(pd.Series(window.predict(decisions), index=decisions.index))
        clim.append(pd.Series(window.base_rate, index=decisions.index))
    probabilities = pd.concat(probs).rename("p")
    climatology = pd.concat(clim).rename("climatology")
    realized = labels.reindex(probabilities.index)
    known = realized.notna()
    p, y, c = probabilities[known].to_numpy(), realized[known].to_numpy(), climatology[known].to_numpy()
    metrics = {
        "Brier": brier(p, y),
        "Brier (climatology)": brier(c, y),
        "Log loss": log_loss(p, y),
        "Log loss (climatology)": log_loss(c, y),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "Out-of-sample predictions": float(len(y)),
        "Base rate (out-of-sample)": float(y.mean()),
    }
    has_skill = metrics["Brier"] < metrics["Brier (climatology)"]
    logger.info(
        "model skill",
        extra={
            "event": "model_skill" if has_skill else "model_no_skill",
            "horizon": horizon,
            "brier": round(metrics["Brier"], 5),
            "brier_climatology": round(metrics["Brier (climatology)"], 5),
        },
    )
    final = fit_window(features.iloc[: len(features) - horizon], labels.iloc[: len(features) - horizon], horizon)
    latest = features.iloc[[-1]]
    return ModelReport(
        horizon=horizon,
        probabilities=probabilities,
        climatology=climatology,
        labels=realized,
        metrics=metrics,
        reliability=reliability_table(p, y),
        has_skill=has_skill,
        latest_probability=float(final.predict(latest)[0]),
        latest_contributions=final.contributions(latest).iloc[0],
        trials=len(HYPER_GRID),
        fold_params=fold_params,
    )
