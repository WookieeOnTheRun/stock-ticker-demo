import numpy as np
import pytest

from stockapp.model.cv import purged_kfold


@pytest.mark.parametrize(("n", "horizon", "k"), [(1000, 21, 5), (500, 5, 4), (257, 21, 3)])
def test_no_training_label_window_overlaps_any_test_index(n, horizon, k):
    for train, test in purged_kfold(n, horizon, k):
        test_set = set(test.tolist())
        for i in train:
            window = range(i, i + horizon + 1)
            assert not test_set.intersection(window), f"train sample {i} labels overlap the test fold"


def test_embargo_after_test_fold_is_respected():
    n, horizon = 1000, 21
    for train, test in purged_kfold(n, horizon, k=5):
        b = test[-1]
        assert not np.any((train > b) & (train <= b + horizon))


def test_folds_cover_every_sample_once_and_keep_most_training_data():
    folds = purged_kfold(1000, 21, k=5)
    tests = np.concatenate([t for _, t in folds])
    assert sorted(tests.tolist()) == list(range(1000))
    for train, test in folds:
        assert not set(train) & set(test)
        assert len(train) >= 1000 - len(test) - 2 * 21 - 1
