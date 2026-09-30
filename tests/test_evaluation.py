from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from market_pipeline.ml.drift import drift_report, psi, reference_profile
from market_pipeline.ml.evaluation import qlike, regression_metrics, walk_forward_folds
from market_pipeline.ml.models import PersistenceRegressor, SmearedLogRegressor


def test_walk_forward_folds_are_ordered_and_purged() -> None:
    dates = pd.bdate_range("2020-01-01", periods=1000)
    folds = walk_forward_folds(dates, n_splits=4, test_days=100, min_train_days=300, embargo_days=5)
    assert len(folds) == 4
    for prev, cur in pairwise(folds):
        assert cur.test_start > prev.test_end  # contiguous, non-overlapping test blocks
    for f in folds:
        # train_end is exclusive: exactly `embargo` sessions sit between train and test.
        gap = dates[(dates >= f.train_end) & (dates < f.test_start)]
        assert len(gap) == 5
    assert folds[-1].test_end == dates[-1]


def test_walk_forward_shrinks_test_blocks_when_history_is_short() -> None:
    dates = pd.bdate_range("2020-01-01", periods=700)
    folds = walk_forward_folds(dates, n_splits=5, test_days=252, min_train_days=300, embargo_days=5)
    assert len(folds) == 5
    with pytest.raises(ValueError, match="not enough history"):
        walk_forward_folds(
            dates[:200], n_splits=5, test_days=20, min_train_days=300, embargo_days=5
        )


def test_qlike_is_zero_for_perfect_and_asymmetric() -> None:
    y = np.array([0.2, 0.3])
    assert qlike(y, y) == pytest.approx(0)
    under, over = qlike(y, y * 0.8), qlike(y, y * 1.2)
    assert under > over > 0  # under-forecasting variance is punished harder


def test_smearing_removes_variance_bias() -> None:
    rng = np.random.default_rng(1)
    n = 20_000
    log_sigma = rng.normal(-1.5, 0.3, n)
    realised_log = log_sigma + rng.normal(0, 0.4, n)  # noisy log-vol proxy
    X = pd.DataFrame({"x": log_sigma})

    model = SmearedLogRegressor(PersistenceRegressor()).fit(X, realised_log)
    pred = model.predict(X)
    # exp(2 * pred) should be unbiased for realised variance; the raw forecast is not.
    assert np.mean(np.exp(2 * realised_log)) / np.mean(np.exp(2 * pred)) == pytest.approx(
        1, abs=0.03
    )
    assert np.mean(np.exp(2 * realised_log)) / np.mean(np.exp(2 * log_sigma)) > 1.3
    assert model.smear_ == pytest.approx(0.4**2, abs=0.02)  # lognormal: 0.5*log E[e^{2e}] = sigma^2


def test_regression_metrics_keys() -> None:
    m = regression_metrics(np.log([0.2, 0.3, 0.4]), np.log([0.25, 0.3, 0.35]))
    assert set(m) == {"rmse_log", "mae_log", "r2_log", "qlike", "bias_log"}


def test_psi_flags_shift_but_not_resample() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"f": rng.normal(0, 1, 5000)})
    profile = reference_profile(ref, ["f"])
    assert psi(profile["f"], rng.normal(0, 1, 2000)) < 0.05
    assert psi(profile["f"], rng.normal(1.5, 1, 2000)) > 0.25
    report = drift_report(profile, pd.DataFrame({"f": rng.normal(1.5, 1, 2000)}))
    assert report.loc[0, "status"] == "alert"
