"""Candidate models, all as sklearn pipelines: raw feature frame in, log-volatility out.

* ``naive``    — persistence: next week's vol = trailing 21-day vol. The bar to clear.
* ``har``      — HAR-RV (Corsi, 2009): a linear model on weekly/monthly/quarterly realised vol.
  It's the standard econometric benchmark for volatility forecasting.
* ``lightgbm`` — gradient-boosted trees over the full feature set, including range-based
  estimators, volume shocks and market-wide volatility.

Each model is fit on log-volatility, and exp(E[log sigma]) under-predicts the variance
level (Jensen's inequality). QLIKE penalises exactly that. So every candidate, the
baseline included, is wrapped in ``SmearedLogRegressor``, which adds Duan's (1983)
smearing correction estimated on a time-ordered holdout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

VOL_FLOOR = 1e-3  # annualised; guards log(0) on halted / flat sessions

VOL_FEATURES = [
    "rv_5d",
    "rv_21d",
    "rv_63d",
    "parkinson_5d",
    "parkinson_21d",
    "market_rv_21d",
]
OTHER_FEATURES = [
    "relative_rv_21d",
    "abs_return_1d",
    "log_return",
    "overnight_gap",
    "momentum_21d",
    "volume_z_21d",
    "day_of_week",
]
ALL_FEATURES = VOL_FEATURES + OTHER_FEATURES
TARGET = "target_rv_fwd"

# Types MLflow's skops serialiser needs allow-listed to load our pipelines (no arbitrary pickle).
SKOPS_TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
    "market_pipeline.ml.models.PersistenceRegressor",
    "market_pipeline.ml.models.SmearedLogRegressor",
    "market_pipeline.ml.models.log_vol",
]


def log_vol(x: Any) -> Any:
    return np.log(np.clip(x, VOL_FLOOR, None))


def log_target(y: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(log_vol(np.asarray(y, dtype=float)))


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """Predicts the first input column unchanged (the forecast is the current value)."""

    def fit(self, X: Any, y: Any = None) -> PersistenceRegressor:
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(X)[:, 0].astype(float)


class SmearedLogRegressor(RegressorMixin, BaseEstimator):
    """Wraps a log-target regressor and calibrates its level for variance forecasts.

    ``fit`` expects rows in time order. The last ``holdout_frac`` of rows is used to
    estimate ``smear_ = 0.5 * log(mean(exp(2 * residual)))``, and the estimator is then
    refit on everything. ``predict`` returns ``log sigma_hat + smear_``, so that
    ``exp(2 * prediction)`` is an unbiased variance forecast.
    """

    def __init__(self, estimator: Any, holdout_frac: float = 0.2) -> None:
        self.estimator = estimator
        self.holdout_frac = holdout_frac

    def fit(self, X: Any, y: Any) -> SmearedLogRegressor:
        y = np.asarray(y, dtype=float)
        cut = int(len(y) * (1 - self.holdout_frac))
        probe = clone(self.estimator).fit(X.iloc[:cut], y[:cut])
        resid = y[cut:] - probe.predict(X.iloc[cut:])
        self.smear_ = float(0.5 * np.log(np.mean(np.exp(2 * resid))))
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator_.predict(X), dtype=float) + self.smear_


def _preprocessor(vol_cols: list[str], other_cols: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = [
        ("log_vol", FunctionTransformer(log_vol, feature_names_out="one-to-one"), vol_cols)
    ]
    if other_cols:
        transformers.append(("raw", "passthrough", other_cols))
    return ColumnTransformer(transformers, verbose_feature_names_out=False).set_output(
        transform="pandas"
    )


def build_naive() -> Pipeline:
    return Pipeline([("pre", _preprocessor(["rv_21d"], [])), ("model", PersistenceRegressor())])


def build_har() -> Pipeline:
    cols = ["rv_5d", "rv_21d", "rv_63d"]
    return Pipeline([("pre", _preprocessor(cols, [])), ("model", LinearRegression())])


def build_lightgbm(params: dict[str, Any] | None = None) -> Pipeline:
    defaults = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "random_state": 42}
    est = LGBMRegressor(**{**defaults, **(params or {}), "verbose": -1})
    return Pipeline([("pre", _preprocessor(VOL_FEATURES, OTHER_FEATURES)), ("model", est)])


def candidate_models(
    lgbm_params: dict[str, Any] | None = None,
) -> dict[str, Callable[[], SmearedLogRegressor]]:
    return {
        "naive": lambda: SmearedLogRegressor(build_naive()),
        "har": lambda: SmearedLogRegressor(build_har()),
        "lightgbm": lambda: SmearedLogRegressor(build_lightgbm(lgbm_params)),
    }


def feature_importance(wrapped: SmearedLogRegressor) -> pd.DataFrame | None:
    pipeline: Pipeline = wrapped.estimator_
    model = pipeline.named_steps["model"]
    if isinstance(model, LGBMRegressor):
        names = pipeline.named_steps["pre"].get_feature_names_out()
        gain = model.booster_.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": names, "gain": gain / gain.sum()})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
    if isinstance(model, LinearRegression):
        names = pipeline.named_steps["pre"].get_feature_names_out()
        return pd.DataFrame({"feature": names, "coef": model.coef_})
    return None
