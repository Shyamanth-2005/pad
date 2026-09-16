"""Soft-voting ensemble over the three trained clinical PAD pipelines.

Each artefact in src/models is a complete scikit-learn Pipeline (imputation +
encoding/scaling + classifier), so raw form values can be handed straight to
predict_proba. The ensemble probability is the unweighted mean of the three -
all three models score within 0.002 ROC-AUC of each other, so weighting them
would be fitting noise.
"""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "src" / "models"
METRICS_CSV = ROOT / "src" / "results" / "model_metrics.csv"

MODEL_FILES = {
    "LogisticRegression": "logistic_regression_clinical.joblib",
    "RandomForest": "random_forest_clinical.joblib",
    "XGBoost": "xgboost_clinical.joblib",
}

NUMERIC_FEATURES = ["Age", "BMI", "HbA1c", "SBP", "DBP",
                    "Total_Cholesterol", "HDL", "TC_HDL_Ratio"]
CATEGORICAL_FEATURES = ["Sex", "Race", "Smoking", "Diabetes",
                        "Medical_History_Hypertension"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FALLBACK_THRESHOLD = 0.5


class PADEnsemble:
    """Loads the three pipelines once and scores one patient at a time."""

    def __init__(self, model_dir: Path = MODEL_DIR, metrics_csv: Path = METRICS_CSV):
        self.models = {}
        missing = []
        for name, filename in MODEL_FILES.items():
            path = model_dir / filename
            if not path.exists():
                missing.append(filename)
                continue
            self.models[name] = joblib.load(path)
        if missing:
            raise FileNotFoundError(
                f"Missing trained model(s) in {model_dir}: {', '.join(missing)}. "
                "Run src/train/train_clinical.py first.")
        self.threshold = self._load_threshold(metrics_csv)

    @staticmethod
    def _load_threshold(metrics_csv: Path) -> float:
        """Mean of the per-model Youden-optimal thresholds from the training run.

        Youden maximises sensitivity + specificity, which is the right operating
        point for a screening tool where a missed case costs more than a recall.
        """
        if not metrics_csv.exists():
            return FALLBACK_THRESHOLD
        metrics = pd.read_csv(metrics_csv)
        youden = metrics[metrics["operating_point"] == "youden"]
        youden = youden[youden["model"].isin(MODEL_FILES)]
        if youden.empty:
            return FALLBACK_THRESHOLD
        return round(float(youden["threshold"].mean()), 4)

    def predict(self, features: dict) -> dict:
        """Score one patient. `features` keys must cover FEATURES; None means missing."""
        frame = pd.DataFrame([{name: features.get(name) for name in FEATURES}])
        frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype("float64")

        per_model = {name: float(model.predict_proba(frame)[0, 1])
                     for name, model in self.models.items()}
        probability = sum(per_model.values()) / len(per_model)
        detected = probability >= self.threshold

        return {
            "pad_detected": bool(detected),
            "probability": round(probability, 4),
            "confidence": round(self._confidence(probability, detected), 4),
            "threshold": self.threshold,
            "per_model": {k: round(v, 4) for k, v in per_model.items()},
        }

    def _confidence(self, probability: float, detected: bool) -> float:
        """How far the score sits from the decision boundary, as 0.5 to 1.0.

        Reporting the raw probability as confidence breaks down at a screening
        threshold: a positive verdict at 0.13 would read as "PAD detected,
        13% confident". Measuring the margin instead puts a score sitting exactly
        on the threshold at 0.5 and one at either extreme at 1.0, which is what
        "confidence in this verdict" should mean. The raw probability is still
        returned separately.
        """
        threshold = self.threshold
        if detected:
            span = 1.0 - threshold
            margin = (probability - threshold) / span if span > 0 else 1.0
        else:
            margin = (threshold - probability) / threshold if threshold > 0 else 1.0
        return 0.5 + 0.5 * min(max(margin, 0.0), 1.0)


_ensemble: PADEnsemble | None = None


def get_ensemble() -> PADEnsemble:
    """Process-wide singleton; the pipelines are read-only after loading."""
    global _ensemble
    if _ensemble is None:
        _ensemble = PADEnsemble()
    return _ensemble
