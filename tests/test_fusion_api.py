import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.prediction.fusion import (FUSION_METRICS_CSV, SENSOR_FEATURES, get_fusion,
                                   stacking_design)

CLINICAL = {
    "Age": 71, "Sex": "Male", "Race": "Non-Hispanic Black", "BMI": 31.2,
    "Smoking": "Current", "Diabetes": "Yes", "HbA1c": 8.4,
    "SBP": 158, "DBP": 88, "Total_Cholesterol": 236, "HDL": 38,
    "Medical_History_Hypertension": "Yes", "ABI": 0.72,
}
# Typical of participants without PAD in NHANES 2003-2004 (feature medians).
ACTIVE = {"Log_Total_Counts": 12.26, "Sedentary_Share": 0.585, "Light_Minutes": 331,
          "MVPA_Minutes": 10.9, "MVPA_Bout_Minutes": 0, "ASTP": 0.275, "SATP": 0.194,
          "Peak30_Counts": 1928, "Longest_Active_Bout": 28.8, "Log_PM_AM_Ratio": 0.25}
# Low, fragmented activity, as seen with PAD.
INACTIVE = {"Log_Total_Counts": 11.2, "Sedentary_Share": 0.76, "Light_Minutes": 190,
            "MVPA_Minutes": 0.5, "MVPA_Bout_Minutes": 0, "ASTP": 0.42, "SATP": 0.12,
            "Peak30_Counts": 700, "Longest_Active_Bout": 12, "Log_PM_AM_Ratio": 0.1}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_stacking_design_ignores_sensor_when_mask_is_zero():
    design = stacking_design([0.2, 0.2], [np.nan, 0.7], [0, 1])
    clinical_logit = np.log(0.2 / 0.8)
    assert design[0].tolist() == pytest.approx([clinical_logit, 0, 0, 0])
    assert design[1].tolist() == pytest.approx(
        [clinical_logit, np.log(0.7 / 0.3), 1, clinical_logit])


def test_without_recording_fusion_preserves_clinical_ranking():
    stack = get_fusion().stacking_model
    clinical = np.linspace(0.01, 0.95, 50)
    fused = stack.predict_proba(stacking_design(clinical, np.nan, np.zeros(50)))[:, 1]
    assert np.all(np.diff(fused) > 0)


def test_threshold_is_the_deployed_row():
    metrics = pd.read_csv(FUSION_METRICS_CSV)
    row = metrics[(metrics.model == "masked_stacking") & (metrics.operating_point == "deployed")]
    assert get_fusion().threshold == pytest.approx(row.threshold.iloc[0], abs=1e-4)
    assert row.sensitivity.iloc[0] >= 0.80


def test_less_activity_raises_sensor_and_fused_risk():
    fusion = get_fusion()
    active = fusion.predict(0.2, ACTIVE)
    inactive = fusion.predict(0.2, INACTIVE)
    assert inactive["sensor_probability"] > active["sensor_probability"]
    assert inactive["probability"] > active["probability"]


def test_clinical_only_request_is_unchanged(client):
    body = client.post("/api/predict", json=CLINICAL).json()
    health = client.get("/api/health").json()
    assert body["modalities"] == ["clinical"]
    assert body["threshold"] == health["threshold"]
    assert body["branch_probabilities"] == {"clinical": body["probability"]}


def test_sensor_request_is_fused(client):
    body = client.post("/api/predict", json={**CLINICAL, "Accelerometry": INACTIVE}).json()
    health = client.get("/api/health").json()
    assert body["modalities"] == ["clinical", "sensor"]
    assert body["threshold"] == health["sensor_threshold"]
    assert set(body["branch_probabilities"]) == {"clinical", "sensor"}
    assert body["pad_detected"] == (body["probability"] >= body["threshold"])
    assert 0.5 <= body["confidence"] <= 1.0


@pytest.mark.parametrize("age", [18, 35, 85])
def test_sensor_branch_accepts_ages_18_to_85(client, age):
    response = client.post("/api/predict",
                           json={**CLINICAL, "Age": age, "Accelerometry": ACTIVE})
    assert response.status_code == 200
    assert response.json()["modalities"] == ["clinical", "sensor"]


@pytest.mark.parametrize("age", [17, 86])
def test_sensor_branch_rejects_ages_outside_18_to_85(client, age):
    response = client.post("/api/predict",
                           json={**CLINICAL, "Age": age, "Accelerometry": ACTIVE})
    assert response.status_code == 422


@pytest.mark.parametrize("change", [
    {"Sedentary_Share": 1.2},
    {"MVPA_Minutes": 5, "MVPA_Bout_Minutes": 10},
])
def test_invalid_accelerometry_is_rejected(client, change):
    response = client.post("/api/predict",
                           json={**CLINICAL, "Accelerometry": {**ACTIVE, **change}})
    assert response.status_code == 422


def test_missing_accelerometry_field_is_rejected(client):
    partial = {name: ACTIVE[name] for name in SENSOR_FEATURES[:-1]}
    response = client.post("/api/predict", json={**CLINICAL, "Accelerometry": partial})
    assert response.status_code == 422
