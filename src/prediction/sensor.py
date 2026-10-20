"""Sensor branch: five models on the ten activity features, combined by weight.

Logistic regression, random forest, XGBoost, LightGBM and CatBoost are each
fitted on the same features. Their probabilities are averaged with non-negative
weights that sum to 1, learned by src/train/train_sensor.py from out-of-fold
validation predictions, so a model that validates poorly gets little say.
Unpickling a fitted ensemble needs xgboost, lightgbm and catboost installed.
"""
import numpy as np
import pandas as pd

# Display order and the keys the fitted ensemble stores its members under.
SENSOR_MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost",
                      "lightgbm", "catboost"]


class SensorEnsemble:
    """Fitted member models and the weights that combine them."""

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
