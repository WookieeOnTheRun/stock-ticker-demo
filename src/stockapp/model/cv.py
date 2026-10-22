"""Purged, embargoed K-fold cross-validation for samples whose forward labels overlap."""

import numpy as np


def purged_kfold(n: int, horizon: int, k: int = 5, embargo: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Purged, embargoed K-fold cross-validation (Lopez de Prado, Advances in Financial Machine Learning, ch. 7) for
    n time-ordered samples whose labels span positions [i, i + horizon]. Test folds are contiguous blocks. For each
    test fold [a, b], training drops every sample whose label window overlaps it (i <= b and i + horizon >= a) and
    an embargo of `embargo` samples after it (default: horizon), since those features overlap the test labels'
    information. Returns (train_positions, test_positions) pairs.
    """
    embargo = horizon if embargo is None else embargo
    positions = np.arange(n)
    folds = []
    for test in np.array_split(positions, k):
        if test.size == 0:
            continue
        a, b = int(test[0]), int(test[-1])
        overlaps = (positions <= b) & (positions + horizon >= a)
        embargoed = (positions > b) & (positions <= b + embargo)
        train = positions[~overlaps & ~embargoed]
        folds.append((train, test))
    return folds
