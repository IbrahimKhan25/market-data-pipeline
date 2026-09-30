"""Time-series cross-validation and forecast loss functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: pd.Timestamp  # exclusive; already purged by the embargo
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive


def walk_forward_folds(
    dates: pd.Series | pd.DatetimeIndex,
    n_splits: int,
    test_days: int,
    min_train_days: int,
    embargo_days: int,
) -> list[Fold]:
    """Expanding-window walk-forward folds over unique trading dates.

    The last ``n_splits * test_days`` dates are cut into consecutive test blocks. Each
    fold trains on everything before its block, minus an embargo of ``embargo_days``
    sessions. The embargo is needed because the label on day t spans t+1..t+h: without
    it, the final training labels would overlap the test period and leak future returns.

    If there isn't enough history, ``test_days`` is shrunk so every fold still gets at
    least ``min_train_days`` of training data.
    """
    unique = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(dates))))
    n = len(unique)
    available = n - min_train_days - embargo_days
    if available < n_splits:
        raise ValueError(
            f"not enough history for {n_splits} folds: {n} dates, "
            f"need > {min_train_days + embargo_days + n_splits}"
        )
    test_days = min(test_days, available // n_splits)
    first_test = n - n_splits * test_days

    folds = []
    for k in range(n_splits):
        start = first_test + k * test_days
        end = start + test_days - 1
        folds.append(
            Fold(
                number=k,
                train_end=unique[start - embargo_days],
                test_start=unique[start],
                test_end=unique[end],
            )
        )
    return folds


def qlike(y_true_vol: np.ndarray, y_pred_vol: np.ndarray) -> float:
    """QLIKE loss on variances (Patton, 2011).

    It stays consistent for ranking volatility forecasts even when the realised-vol
    proxy is noisy, and it penalises under-prediction more than over-prediction.
    """
    ratio = (np.asarray(y_true_vol) ** 2) / (np.asarray(y_pred_vol) ** 2)
    return float(np.mean(ratio - np.log(ratio) - 1))


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred_log) - np.asarray(y_true_log)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true_log - np.mean(y_true_log)) ** 2))
    return {
        "rmse_log": float(np.sqrt(np.mean(err**2))),
        "mae_log": float(np.mean(np.abs(err))),
        "r2_log": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "qlike": qlike(np.exp(y_true_log), np.exp(y_pred_log)),
        "bias_log": float(np.mean(err)),
    }
