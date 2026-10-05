from pathlib import Path

import pytest

needed = [Path("models/model.joblib"), Path("models/quantile_models.joblib")]
if not all(path.exists() for path in needed):
    pytest.skip("trained models arent stored in the repo", allow_module_level=True)

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_a_full_day():
    response = client.get("/predict", params={"day": "2026-08-26"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecast"]) == 48
    assert body["in_training_data"] is False
    for row in body["forecast"]:
        assert row["low"] <= row["forecast"] <= row["high"]

def test_predict_unknown_date_gives_404():
    response = client.get("/predict", params={"day": "2030-01-01"})
    assert response.status_code == 404

def test_predict_invalid_date_gives_422():
    response = client.get("/predict", params={"day": "hello"})
    assert response.status_code == 422