# tests/test_api.py

"""Minimal smoke tests — crashes, response shape, validation."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.model_loader import ModelArtifacts

FEATURE_COLS = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "EXT_SOURCE_2"]


@pytest.fixture
def mock_artifacts():
    from unittest.mock import MagicMock

    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.85, 0.15]])

    return ModelArtifacts(
        model=model,
        encoders={},
        feature_cols=FEATURE_COLS,
        explainer=None,
        model_version="v1-test",
        cv_auc=0.787,
    )


@pytest.fixture
def client(mock_artifacts, monkeypatch):
    import api.main as main_module
    import api.predictor as predictor_module

    # Bypass full preprocessing — return a minimal aligned DataFrame
    def mock_preprocess(request, artifacts):
        return pd.DataFrame([[0.0] * len(FEATURE_COLS)], columns=FEATURE_COLS)

    monkeypatch.setattr(predictor_module, "_preprocess", mock_preprocess)
    main_module._artifacts = mock_artifacts
    return TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "AMT_CREDIT": 450000.0,
        "AMT_INCOME_TOTAL": 180000.0,
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "DAYS_BIRTH": -12000,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "EXT_SOURCE_2": 0.6,
    }


def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_predict_returns_200(client, valid_payload):
    assert client.post("/predict", json=valid_payload).status_code == 200


def test_predict_probability_in_range(client, valid_payload):
    data = client.post("/predict", json=valid_payload).json()
    assert 0.0 <= data["default_probability"] <= 1.0


def test_predict_invalid_input_returns_422(client, valid_payload):
    valid_payload["AMT_CREDIT"] = -1000
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_explain_without_explainer_returns_503(client, valid_payload):
    assert client.post("/explain", json=valid_payload).status_code == 503