"""Sensor branch and availability-masked fusion layer (report Sections 3.6-3.7).

The sensor branch is a standardised logistic regression on ten free-living
activity features from a seven-day hip accelerometer, giving p_s. The fusion
layer is a logistic stacking model over the branch logits and an availability
mask m (1 when a valid recording exists):

    logit p_f = b0 + b1*l_c + b2*m*l_s + b3*m + b4*m*l_c          (3.2)

With m = 0 the fused score is a monotone recalibration of the clinical score,
so a patient without a recording keeps their clinical ranking. Both models are
fitted by src/train/train_sensor.py on NHANES 2003-2004; the clinical score l_c
comes from the frozen clinical ensemble in src/prediction/ensemble.py.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.prediction.ensemble import decision_confidence

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "src" / "models"
FUSION_METRICS_CSV = ROOT / "src" / "results" / "sensor" / "fusion_metrics.csv"

SENSOR_MODEL_FILE = "sensor_logistic_regression.joblib"
FUSION_MODEL_FILE = "fusion_stacking.joblib"

# Table 3.2, in the order the sensor pipeline expects.
SENSOR_VOLUME_FEATURES = ["Log_Total_Counts", "Sedentary_Share", "Light_Minutes",
                          "MVPA_Minutes", "MVPA_Bout_Minutes"]
SENSOR_PATTERN_FEATURES = ["ASTP", "SATP", "Peak30_Counts", "Longest_Active_Bout",
                           "Log_PM_AM_Ratio"]
SENSOR_FEATURES = SENSOR_VOLUME_FEATURES + SENSOR_PATTERN_FEATURES

# Probabilities are clipped before the logit so a saturated score stays finite.
_PROBABILITY_EPS = 1e-6


def logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), _PROBABILITY_EPS,
                1 - _PROBABILITY_EPS)
    return np.log(p / (1 - p))


def stacking_design(clinical_probability, sensor_probability, mask) -> np.ndarray:
    """Columns of eq. (3.2): l_c, m*l_s, m, m*l_c.

    Where m = 0 the sensor probability is ignored (it may be NaN), so a missing
    recording never needs imputing.
    """
    mask = np.asarray(mask, dtype=float)
    clinical_logit = logit(clinical_probability)
    sensor_logit = np.where(mask == 1, logit(np.nan_to_num(sensor_probability, nan=0.5)), 0.0)
    return np.column_stack([clinical_logit, mask * sensor_logit, mask,
                            mask * clinical_logit])


class SensorFusion:
    """Loads the sensor model and stacking layer once; fuses one patient at a time."""

    def __init__(self, model_dir: Path = MODEL_DIR,
                 metrics_csv: Path = FUSION_METRICS_CSV):
        paths = [model_dir / SENSOR_MODEL_FILE, model_dir / FUSION_MODEL_FILE, metrics_csv]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing sensor artefact(s): {', '.join(missing)}. "
                "Run src/train/train_sensor.py first.")
        self.sensor_model = joblib.load(paths[0])
        self.stacking_model = joblib.load(paths[1])
        self.threshold = self._load_threshold(metrics_csv)

    @staticmethod
    def _load_threshold(metrics_csv: Path) -> float:
        """The deployed operating point written by the training run.

        No fallback: a 0.5 default would detect almost nothing at a 10% prevalence.
        """
        metrics = pd.read_csv(metrics_csv)
        deployed = metrics.loc[(metrics["model"] == "masked_stacking")
                               & (metrics["operating_point"] == "deployed"), "threshold"]
        if deployed.empty:
            raise ValueError(f"No deployed masked_stacking row in {metrics_csv}.")
        return round(float(deployed.iloc[0]), 4)

    def predict(self, clinical_probability: float, sensor_features: dict) -> dict:
        """Fuse a clinical ensemble probability with one valid recording (m = 1)."""
        frame = pd.DataFrame([{name: sensor_features[name] for name in SENSOR_FEATURES}],
                             dtype="float64")
        sensor_probability = float(self.sensor_model.predict_proba(frame)[0, 1])
        design = stacking_design([clinical_probability], [sensor_probability], [1])
        probability = float(self.stacking_model.predict_proba(design)[0, 1])
        return {
            "pad_detected": bool(probability >= self.threshold),
            "probability": round(probability, 4),
            "confidence": round(decision_confidence(probability, self.threshold), 4),
            "threshold": self.threshold,
            "sensor_probability": round(sensor_probability, 4),
        }


_fusion: SensorFusion | None = None


def get_fusion() -> SensorFusion:
    """Process-wide singleton; the models are read-only after loading."""
    global _fusion
    if _fusion is None:
        _fusion = SensorFusion()
    return _fusion
