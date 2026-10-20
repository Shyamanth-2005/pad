"""Real-cohort clinical branch, sensor branch and availability-masked fusion
(report Sections 3.6-3.7).

Three branch scores feed the fusion layer:

    p_c  frozen clinical ensemble (src/prediction/ensemble.py), trained on the
         augmented 50,000-record cohort
    p_r  logistic regression on the same clinical features, fitted on the real
         NHANES 2003-2004 cohort
    p_s  weighted ensemble of five models (src/prediction/sensor.py) on ten
         free-living activity features from a seven-day hip accelerometer,
         present only with a valid recording

The fusion layer is a logistic stacking model over the branch logits and an
availability mask m (1 when a valid recording exists):

    logit p_f = b0 + b1*l_c + b2*l_r + b3*m*l_s + b4*m + b5*m*l_c + b6*m*l_r

Every verdict goes through it. With m = 0 it combines and recalibrates the two
clinical branches, so missing accelerometry never needs imputing. All three
fitted artefacts come from src/train/train_sensor.py.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.prediction.ensemble import clinical_frame, decision_confidence

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "src" / "models"
FUSION_METRICS_CSV = ROOT / "src" / "results" / "sensor" / "fusion_metrics.csv"

COHORT_MODEL_FILE = "nhanes_clinical_logistic_regression.joblib"
SENSOR_MODEL_FILE = "sensor_ensemble.joblib"
FUSION_MODEL_FILE = "fusion_stacking.joblib"

# Table 3.2, in the order the sensor pipeline expects.
SENSOR_VOLUME_FEATURES = ["Log_Total_Counts", "Sedentary_Share", "Light_Minutes",
                          "MVPA_Minutes", "MVPA_Bout_Minutes"]
SENSOR_PATTERN_FEATURES = ["ASTP", "SATP", "Peak30_Counts", "Longest_Active_Bout",
                           "Log_PM_AM_Ratio"]
SENSOR_FEATURES = SENSOR_VOLUME_FEATURES + SENSOR_PATTERN_FEATURES

# The deployed operating points written by the training run, one per mode.
SENSOR_THRESHOLD_MODEL = "masked_stacking"
CLINICAL_THRESHOLD_MODEL = "clinical_only"

# Probabilities are clipped before the logit so a saturated score stays finite.
_PROBABILITY_EPS = 1e-6


def logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), _PROBABILITY_EPS,
                1 - _PROBABILITY_EPS)
    return np.log(p / (1 - p))


def stacking_design(clinical_probability, cohort_probability, sensor_probability,
                    mask) -> np.ndarray:
    """Columns of the fusion equation: l_c, l_r, m*l_s, m, m*l_c, m*l_r.

    Where m = 0 the sensor probability is ignored (it may be NaN), so a missing
    recording never needs imputing.
    """
    mask = np.asarray(mask, dtype=float)
    clinical_logit = logit(clinical_probability)
    cohort_logit = logit(cohort_probability)
    sensor_logit = np.where(mask == 1, logit(np.nan_to_num(sensor_probability, nan=0.5)), 0.0)
    return np.column_stack([clinical_logit, cohort_logit, mask * sensor_logit, mask,
                            mask * clinical_logit, mask * cohort_logit])


class SensorFusion:
    """Loads the NHANES clinical model, sensor ensemble and stacking layer once."""

    def __init__(self, model_dir: Path = MODEL_DIR,
                 metrics_csv: Path = FUSION_METRICS_CSV):
        paths = [model_dir / COHORT_MODEL_FILE, model_dir / SENSOR_MODEL_FILE,
                 model_dir / FUSION_MODEL_FILE, metrics_csv]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing fusion artefact(s): {', '.join(missing)}. "
                "Run src/train/train_sensor.py first.")
        self.cohort_model = joblib.load(paths[0])
        self.sensor_model = joblib.load(paths[1])
        self.stacking_model = joblib.load(paths[2])
        self.clinical_threshold = self._load_threshold(metrics_csv, CLINICAL_THRESHOLD_MODEL)
        self.sensor_threshold = self._load_threshold(metrics_csv, SENSOR_THRESHOLD_MODEL)

    @staticmethod
    def _load_threshold(metrics_csv: Path, model: str) -> float:
        """No fallback: a 0.5 default would detect almost nothing at a 10% prevalence."""
        metrics = pd.read_csv(metrics_csv)
        deployed = metrics.loc[(metrics["model"] == model)
                               & (metrics["operating_point"] == "deployed"), "threshold"]
        if deployed.empty:
            raise ValueError(f"No deployed {model} row in {metrics_csv}.")
        return round(float(deployed.iloc[0]), 4)

    def predict(self, clinical_probability: float, clinical_features: dict,
                sensor_features: dict | None = None) -> dict:
        """Fuse one patient's branch scores; `sensor_features` None means m = 0."""
        cohort_probability = float(
            self.cohort_model.predict_proba(clinical_frame(clinical_features))[0, 1])
        if sensor_features is None:
            sensor_probability, available, threshold = np.nan, 0, self.clinical_threshold
        else:
            frame = pd.DataFrame([{name: sensor_features[name] for name in SENSOR_FEATURES}],
                                 dtype="float64")
            sensor_probability = float(self.sensor_model.predict_proba(frame)[0, 1])
            available, threshold = 1, self.sensor_threshold
        design = stacking_design([clinical_probability], [cohort_probability],
                                 [sensor_probability], [available])
        probability = float(self.stacking_model.predict_proba(design)[0, 1])
        result = {
            "pad_detected": bool(probability >= threshold),
            "probability": round(probability, 4),
            "confidence": round(decision_confidence(probability, threshold), 4),
            "threshold": threshold,
            "cohort_probability": round(cohort_probability, 4),
        }
        if available:
            result["sensor_probability"] = round(sensor_probability, 4)
        return result


_fusion: SensorFusion | None = None


def get_fusion() -> SensorFusion:
    """Process-wide singleton; the models are read-only after loading."""
    global _fusion
    if _fusion is None:
        _fusion = SensorFusion()
    return _fusion
