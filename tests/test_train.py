import json

import mlflow
import pytest

from market_pipeline.ml.models import ALL_FEATURES
from market_pipeline.ml.train import REGISTERED_MODEL, load_model, train


def test_champion_beats_naive_and_is_promoted(built) -> None:  # type: ignore[no-untyped-def]
    r = built.trained
    assert set(r.summary.index) == {"naive", "har", "lightgbm"}
    assert r.champion != "naive"
    assert r.promoted
    assert r.summary.loc[r.champion, "qlike_improvement_vs_naive"] >= 0.05
    assert set(r.folds["fold"]) == {0, 1, 2}


def test_promoted_artifacts_and_card(built) -> None:  # type: ignore[no-untyped-def]
    _, card = load_model(built.settings)
    assert card["champion"] == built.trained.champion
    assert card["features"] == ALL_FEATURES
    assert set(card["reference_profile"]) == set(ALL_FEATURES)
    assert card["mlflow_run_id"] == built.trained.mlflow_run_id
    json.dumps(card)  # serialisable


def test_mlflow_run_and_registry(built) -> None:  # type: ignore[no-untyped-def]
    mlflow.set_tracking_uri(built.settings.ml.mlflow_tracking_uri)
    run = mlflow.get_run(built.trained.mlflow_run_id)
    assert run.data.tags["champion"] == built.trained.champion
    assert f"{built.trained.champion}.qlike_mean" in run.data.metrics
    reloaded = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@champion")
    assert hasattr(reloaded, "smear_")


def test_gate_blocks_promotion_and_keeps_current_model(built) -> None:  # type: ignore[no-untyped-def]
    strict = built.settings.model_copy(
        update={"ml": built.settings.ml.model_copy(update={"min_improvement_vs_naive": 0.99})}
    )
    before = (built.settings.paths.models_dir / "model_card.json").read_text()
    result = train(strict, log_mlflow=False)
    assert not result.promoted
    assert (built.settings.paths.models_dir / "model_card.json").read_text() == before


def test_train_without_data_fails_clearly(settings) -> None:  # type: ignore[no-untyped-def]
    from market_pipeline.warehouse import connect

    settings.paths.warehouse.parent.mkdir(parents=True)
    with connect(settings, read_only=False) as con:
        con.execute("create schema marts")
        con.execute(
            "create table marts.features_volatility (ticker varchar, date date, is_labeled boolean)"
        )
    with pytest.raises(RuntimeError, match="no labelled feature rows"):
        train(settings, log_mlflow=False)
