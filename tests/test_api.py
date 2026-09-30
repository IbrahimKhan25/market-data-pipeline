import pytest
from fastapi.testclient import TestClient

from market_pipeline.api.app import create_app


@pytest.fixture(scope="module")
def client(built):  # type: ignore[no-untyped-def]
    with TestClient(create_app(built.settings)) as c:
        yield c


FEATURES = {
    "rv_5d": 0.2,
    "rv_21d": 0.25,
    "rv_63d": 0.3,
    "parkinson_5d": 0.2,
    "parkinson_21d": 0.22,
    "market_rv_21d": 0.2,
    "relative_rv_21d": 1.25,
    "abs_return_1d": 0.01,
    "log_return": -0.01,
    "overnight_gap": 0.001,
    "momentum_21d": 0.02,
    "volume_z_21d": 0.5,
    "day_of_week": 3,
}


def test_health_and_model_card(client, built) -> None:  # type: ignore[no-untyped-def]
    assert client.get("/health").json()["model_loaded"] is True
    card = client.get("/model").json()
    assert card["champion"] == built.trained.champion
    assert "reference_profile" not in card


def test_forecasts_for_all_and_one_ticker(client) -> None:  # type: ignore[no-untyped-def]
    all_ = client.get("/forecasts").json()
    assert {f["ticker"] for f in all_} == {"AAA", "BBB", "CCC", "DDD"}
    one = client.get("/forecasts/bbb")
    assert one.status_code == 200 and one.json()["ticker"] == "BBB"
    assert client.get("/forecasts/NOPE").status_code == 404


def test_predict_endpoint_validates_input(client) -> None:  # type: ignore[no-untyped-def]
    ok = client.post("/predict", json=FEATURES)
    assert ok.status_code == 200 and 0 < ok.json()["forecast_vol"] < 5
    assert client.post("/predict", json={**FEATURES, "rv_5d": -1}).status_code == 422


def test_503_without_model(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/forecasts").status_code == 503
