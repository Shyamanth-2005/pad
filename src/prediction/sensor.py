"""Sensor branch: five models on the ten activity features, combined by weight.

Logistic regression, random forest, XGBoost, LightGBM and CatBoost are each
fitted on the same features and saved to their own file in src/models. At
prediction time each one scores the patient, and their probabilities are
averaged with non-negative weights that sum to 1. src/train/train_sensor.py
learns the weights from out-of-fold validation predictions, so a model that
validates poorly gets little say, and saves them beside the models.
Loading the models needs xgboost, lightgbm and catboost installed.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Display order and the keys the fitted ensemble stores its members under.
SENSOR_MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost",
                      "lightgbm", "catboost"]
SENSOR_MODEL_FILES = {name: f"sensor_{name}.joblib" for name in SENSOR_MODEL_NAMES}
SENSOR_WEIGHTS_FILE = "sensor_ensemble_weights.json"


class SensorEnsemble:
    """Fitted member models and the weights that combine their decisions."""

    def __init__(self, models: dict, weights: dict):
        if set(models) != set(weights):
            raise ValueError("Every sensor model needs exactly one weight.")
        total = sum(weights.values())
        if total <= 0 or min(weights.values()) < 0:
            raise ValueError("Sensor weights must be non-negative and not all zero.")
        self.models = models
        self.weights = {name: weight / total for name, weight in weights.items()}

    def member_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """Each member's probability of PAD, one column per model."""
        return pd.DataFrame({name: model.predict_proba(features)[:, 1]
                             for name, model in self.models.items()},
                            index=features.index)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Weighted average, in the two-column shape scikit-learn models return."""
        members = self.member_probabilities(features)
        weights = np.array([self.weights[name] for name in members.columns])
        positive = members.to_numpy() @ weights
        return np.column_stack([1 - positive, positive])


def save_sensor_ensemble(ensemble: SensorEnsemble, model_dir: Path) -> None:
    """One file per fitted model, plus the weights as JSON."""
    for name, model in ensemble.models.items():
        joblib.dump(model, model_dir / SENSOR_MODEL_FILES[name], compress=3)
    (model_dir / SENSOR_WEIGHTS_FILE).write_text(
        json.dumps({name: float(weight) for name, weight in ensemble.weights.items()},
                   indent=2), encoding="utf-8")


def load_sensor_ensemble(model_dir: Path) -> SensorEnsemble:
    """Load the five saved models and their weights into one ensemble."""
    files = [*SENSOR_MODEL_FILES.values(), SENSOR_WEIGHTS_FILE]
    missing = [name for name in files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing sensor artefact(s): {', '.join(missing)}. "
            "Run src/train/train_sensor.py first.")
    models = {name: joblib.load(model_dir / SENSOR_MODEL_FILES[name])
              for name in SENSOR_MODEL_NAMES}
    weights = json.loads((model_dir / SENSOR_WEIGHTS_FILE).read_text(encoding="utf-8"))
    return SensorEnsemble(models, weights)
