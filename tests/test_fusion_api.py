import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.prediction.ensemble import clinical_frame
from src.prediction.sensor import SENSOR_MODEL_NAMES, SensorEnsemble
from src.prediction.fusion import (FUSION_METRICS_CSV, SENSOR_FEATURES, get_fusion,
                                   stacking_design)

CLINICAL = {
    "Age": 71, "Sex": "Male", "Race": "Non-Hispanic Black", "BMI": 31.2,
    "Smoking": "Current", "Diabetes": "Yes", "HbA1c": 8.4,
    "SBP": 158, "DBP": 88, "Total_Cholesterol": 236, "HDL": 38,
    "Medical_History_Hypertension": "Yes", "ABI": 0.72,
}
LOW_RISK = {
    "Age": 45, "Sex": "Female", "Race": "Non-Hispanic White", "BMI": 24.0,
    "Smoking": "Never", "Diabetes": "No", "HbA1c": 5.2,
    "SBP": 112, "DBP": 70, "Total_Cholesterol": 180, "HDL": 65,
    "Medical_History_Hypertension": "No", "ABI": 1.10,
}
# The clinical feature dict the fusion layer takes: the record minus the ABI.
FEATURES = {**{k: v for k, v in CLINICAL.items() if k != "ABI"},
            "TC_HDL_Ratio": round(236 / 38, 2)}
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
    design = stacking_design([0.2, 0.2], [0.3, 0.3], [np.nan, 0.7], [0, 1])
    clinical_logit, cohort_logit = np.log(0.2 / 0.8), np.log(0.3 / 0.7)
    assert design[0].tolist() == pytest.approx([clinical_logit, cohort_logit, 0, 0, 0, 0])
    assert design[1].tolist() == pytest.approx(
        [clinical_logit, cohort_logit, np.log(0.7 / 0.3), 1, clinical_logit, cohort_logit])


@pytest.mark.parametrize("model, subset", [("masked_stacking", "valid_recording"),
                                           ("clinical_only", "all_participants")])
def test_thresholds_are_the_deployed_rows(model, subset):
    metrics = pd.read_csv(FUSION_METRICS_CSV)
    row = metrics[(metrics.model == model) & (metrics.subset == subset)
                  & (metrics.operating_point == "deployed")]
    fusion = get_fusion()
    loaded = fusion.sensor_threshold if model == "masked_stacking" else fusion.clinical_threshold
    assert loaded == pytest.approx(row.threshold.iloc[0], abs=1e-4)
    assert row.sensitivity.iloc[0] >= 0.80


def test_sensor_score_is_the_weighted_average_of_all_five_models():
    sensor = get_fusion().sensor_model
    assert list(sensor.models) == SENSOR_MODEL_NAMES
    assert sum(sensor.weights.values()) == pytest.approx(1)
    assert min(sensor.weights.values()) >= 0
    frame = pd.DataFrame([ACTIVE])[SENSOR_FEATURES]
    members = sensor.member_probabilities(frame).iloc[0]
    expected = sum(sensor.weights[name] * members[name] for name in SENSOR_MODEL_NAMES)
    assert sensor.predict_proba(frame)[0, 1] == pytest.approx(expected)


@pytest.mark.parametrize("weights", [{"a": 1.0}, {"a": -0.5, "b": 1.5}, {"a": 0.0, "b": 0.0}])
def test_sensor_ensemble_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        SensorEnsemble({"a": None, "b": None}, weights)


def test_less_activity_raises_sensor_and_fused_risk():
    fusion = get_fusion()
    active = fusion.predict(0.2, FEATURES, ACTIVE)
    inactive = fusion.predict(0.2, FEATURES, INACTIVE)
    assert inactive["sensor_probability"] > active["sensor_probability"]
    assert inactive["probability"] > active["probability"]


def test_clinical_only_request_uses_both_clinical_branches(client):
    body = client.post("/api/predict", json=CLINICAL).json()
    health = client.get("/api/health").json()
    assert body["modalities"] == ["clinical"]
    assert body["threshold"] == health["threshold"]
    assert set(body["branch_probabilities"]) == {"clinical", "nhanes"}
    assert body["pad_detected"] == (body["probability"] >= body["threshold"])


def test_clinical_only_ranks_higher_risk_records_higher(client):
    high = client.post("/api/predict", json=CLINICAL).json()
    low = client.post("/api/predict", json=LOW_RISK).json()
    assert high["probability"] > low["probability"]
    assert high["pad_detected"] and not low["pad_detected"]


def test_clinical_frame_marks_unknown_values_as_nan():
    # The imputers only recognise NaN; a None would reach the one-hot encoder
    # and be read as its dropped reference level.
    frame = clinical_frame({**FEATURES, "Smoking": None, "HbA1c": None})
    assert np.isnan(frame["Smoking"].iloc[0])
    assert np.isnan(frame["HbA1c"].iloc[0])


def test_unknown_smoking_is_imputed_not_read_as_current(client):
    current = client.post("/api/predict", json={**CLINICAL, "Smoking": "Current"}).json()
    unknown = client.post("/api/predict", json={**CLINICAL, "Smoking": None}).json()
    assert unknown["branch_probabilities"]["nhanes"] < current["branch_probabilities"]["nhanes"]
    assert unknown["branch_probabilities"]["clinical"] < current["branch_probabilities"]["clinical"]


def test_sensor_request_is_fused(client):
    body = client.post("/api/predict", json={**CLINICAL, "Accelerometry": INACTIVE}).json()
    health = client.get("/api/health").json()
    assert body["modalities"] == ["clinical", "sensor"]
    assert body["threshold"] == health["sensor_threshold"]
    assert set(body["branch_probabilities"]) == {"clinical", "nhanes", "sensor"}
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
