"""Train step: walk-forward CV of every candidate, champion selection, promotion gate, MLflow.

Champion selection uses QLIKE, and promotion requires beating the naive baseline by a margin.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from market_pipeline.config import Settings
from market_pipeline.ml.drift import reference_profile
from market_pipeline.ml.evaluation import regression_metrics, walk_forward_folds
from market_pipeline.ml.models import (
    ALL_FEATURES,
    SKOPS_TRUSTED_TYPES,
    TARGET,
    candidate_models,
    feature_importance,
    log_target,
)
from market_pipeline.warehouse import connect

log = logging.getLogger(__name__)

MODEL_FILE = "volatility_model.joblib"
CARD_FILE = "model_card.json"
REGISTERED_MODEL = "volatility-forecaster"
SELECTION_METRIC = "qlike"


@dataclass
class TrainResult:
    version: str
    champion: str
    promoted: bool
    summary: pd.DataFrame
    folds: pd.DataFrame
    card: dict[str, Any] = field(default_factory=dict)
    mlflow_run_id: str | None = None


def load_features(settings: Settings, labeled_only: bool = True) -> pd.DataFrame:
    where = "where is_labeled" if labeled_only else ""
    with connect(settings) as con:
        df = con.sql(f"select * from marts.features_volatility {where} order by date, ticker").df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def cross_validate(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    ml = settings.ml
    folds = walk_forward_folds(
        df["date"], ml.n_splits, ml.test_days, ml.min_train_days, embargo_days=ml.horizon_days
    )
    y_all = log_target(df[TARGET])
    rows, preds = [], []
    for fold in folds:
        train = (df["date"] < fold.train_end).to_numpy()
        test = ((df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)).to_numpy()
        for name, build in candidate_models(ml.lgbm_params).items():
            model = build().fit(df.loc[train, ALL_FEATURES], y_all[train])
            y_hat = model.predict(df.loc[test, ALL_FEATURES])
            metrics = regression_metrics(y_all[test], y_hat)
            rows.append(
                {
                    "fold": fold.number,
                    "model": name,
                    "train_rows": int(train.sum()),
                    "test_rows": int(test.sum()),
                    "test_start": fold.test_start.date(),
                    "test_end": fold.test_end.date(),
                    **metrics,
                }
            )
            preds.append(
                pd.DataFrame(
                    {
                        "fold": fold.number,
                        "model": name,
                        "ticker": df.loc[test, "ticker"].to_numpy(),
                        "date": df.loc[test, "date"].to_numpy(),
                        "y_true_log": y_all[test],
                        "y_pred_log": y_hat,
                    }
                )
            )
        log.info(
            "fold %d: test %s..%s done", fold.number, fold.test_start.date(), fold.test_end.date()
        )
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarise(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["rmse_log", "mae_log", "r2_log", "qlike", "bias_log"]
    summary = fold_metrics.groupby("model")[metric_cols].agg(["mean", "std"])
    flat = cast(list[tuple[str, str]], list(summary.columns))
    summary.columns = [f"{metric}_{stat}" for metric, stat in flat]
    for metric in ("qlike", "rmse_log"):
        baseline = cast(float, summary.loc["naive", f"{metric}_mean"])
        summary[f"{metric}_improvement_vs_naive"] = 1 - summary[f"{metric}_mean"] / baseline
    return summary.sort_values(f"{SELECTION_METRIC}_mean")


def _log_to_mlflow(
    settings: Settings,
    result: TrainResult,
    oos: pd.DataFrame,
    pipeline: Any,
    importance: pd.DataFrame | None,
    sample: pd.DataFrame,
) -> str:
    import mlflow
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(settings.ml.mlflow_tracking_uri)
    mlflow.set_experiment(settings.ml.mlflow_experiment)
    with mlflow.start_run(run_name=f"train-{result.version}") as run:
        mlflow.set_tags({"champion": result.champion, "promoted": str(result.promoted)})
        mlflow.log_params(
            {
                "horizon_days": settings.ml.horizon_days,
                "n_splits": settings.ml.n_splits,
                "test_days": settings.ml.test_days,
                "selection_metric": SELECTION_METRIC,
                "n_tickers": result.card["data"]["n_tickers"],
                "train_rows": result.card["data"]["rows"],
                "data_start": result.card["data"]["start"],
                "data_end": result.card["data"]["end"],
                **{f"lgbm_{k}": v for k, v in settings.ml.lgbm_params.items()},
            }
        )
        for model, row in result.summary.iterrows():
            mlflow.log_metrics({f"{model}.{k}": float(v) for k, v in row.items() if pd.notna(v)})
        for _, row in result.folds.iterrows():
            mlflow.log_metric(f"{row['model']}.fold_qlike", row["qlike"], step=int(row["fold"]))

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            result.folds.to_csv(t / "cv_folds.csv", index=False)
            result.summary.to_csv(t / "cv_summary.csv")
            oos.to_parquet(t / "oos_predictions.parquet", index=False)
            if importance is not None:
                importance.to_csv(t / "feature_importance.csv", index=False)
            (t / CARD_FILE).write_text(json.dumps(_card_without_profile(result.card), indent=2))
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        if result.promoted:
            import mlflow.sklearn
            from mlflow import MlflowClient

            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=infer_signature(sample, pipeline.predict(sample)),
                input_example=sample.head(3),
                registered_model_name=REGISTERED_MODEL,
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            if info.registered_model_version is not None:
                MlflowClient().set_registered_model_alias(
                    REGISTERED_MODEL, "champion", str(info.registered_model_version)
                )
        return str(run.info.run_id)


def _card_without_profile(card: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in card.items() if k != "reference_profile"}


def train(settings: Settings, log_mlflow: bool = True) -> TrainResult:
    df = load_features(settings, labeled_only=True)
    if df.empty:
        raise RuntimeError("no labelled feature rows; run ingest + transform first")
    log.info(
        "training on %d rows, %d tickers, %s..%s",
        len(df),
        df["ticker"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )

    fold_metrics, oos = cross_validate(df, settings)
    summary = summarise(fold_metrics)
    champion = str(summary.index[0])
    improvement = float(
        cast(float, summary.loc[champion, f"{SELECTION_METRIC}_improvement_vs_naive"])
    )
    promoted = champion != "naive" and improvement >= settings.ml.min_improvement_vs_naive

    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    pipeline = candidate_models(settings.ml.lgbm_params)[champion]().fit(
        df[ALL_FEATURES], log_target(df[TARGET])
    )
    importance = feature_importance(pipeline)

    card: dict[str, Any] = {
        "version": version,
        "champion": champion,
        "promoted": promoted,
        "selection_metric": SELECTION_METRIC,
        "horizon_days": settings.ml.horizon_days,
        "target": f"annualised realised volatility over next {settings.ml.horizon_days} sessions",
        "features": ALL_FEATURES,
        "cv": {
            model: {k: round(float(v), 5) for k, v in row.items()}
            for model, row in summary.iterrows()
        },
        "data": {
            "rows": len(df),
            "n_tickers": int(df["ticker"].nunique()),
            "tickers": sorted(df["ticker"].unique().tolist()),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
        "gate": {
            "min_improvement_vs_naive": settings.ml.min_improvement_vs_naive,
            "champion_improvement_vs_naive": round(improvement, 4),
        },
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference_profile": reference_profile(df, ALL_FEATURES),
    }
    result = TrainResult(version, champion, promoted, summary, fold_metrics, card)

    if promoted:
        models_dir = settings.paths.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, models_dir / MODEL_FILE)
        log.info(
            "promoted %s (%s): %s %.1f%% better than naive",
            champion,
            version,
            SELECTION_METRIC,
            100 * improvement,
        )
    else:
        log.warning(
            "champion %s did not clear the gate (%.1f%% vs naive < %.1f%%); "
            "keeping the current production model",
            champion,
            100 * improvement,
            100 * settings.ml.min_improvement_vs_naive,
        )

    if log_mlflow:
        sample = df[ALL_FEATURES].tail(200).astype("float64")
        result.mlflow_run_id = _log_to_mlflow(settings, result, oos, pipeline, importance, sample)
        card["mlflow_run_id"] = result.mlflow_run_id

    if promoted:
        # Written last so the card on disk always matches the promoted model file.
        (settings.paths.models_dir / CARD_FILE).write_text(json.dumps(card, indent=2))
    return result


def load_model(settings: Settings) -> tuple[Any, dict[str, Any]]:
    models_dir = settings.paths.models_dir
    model_path, card_path = models_dir / MODEL_FILE, models_dir / CARD_FILE
    if not model_path.exists() or not card_path.exists():
        raise FileNotFoundError(f"no promoted model in {models_dir}; run `market-pipeline train`")
    return joblib.load(model_path), json.loads(card_path.read_text())


def predict_vol(pipeline: Any, features: pd.DataFrame) -> np.ndarray:
    return np.exp(pipeline.predict(features[ALL_FEATURES]))
