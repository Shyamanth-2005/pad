"""
Real-cohort clinical branch, sensor branch and availability-masked fusion
(report Sections 3.6-3.8, Alg. 4.3).

Data:   PAD_nhanes_2003_2004_sensor.csv, built by build_sensor_dataset.py
        (NHANES 2003-2004, adults 40+, PAD = minimum ABI <= 0.90).
Models: NHANES clinical - logistic regression on the 13 clinical features, fitted
                          on this real cohort (the frozen ensemble is fitted on
                          augmented data and ranks real patients worse)
        sensor branch   - standardised logistic regression on ten activity features
        fusion layer    - logistic stacking on [l_c, l_r, m*l_s, m, m*l_c, m*l_r]
        comparators     - the same stacking without the sensor term (clinical
                          branches recalibrated per mask pattern), and an
                          early-fusion logistic regression on all 23 inputs + mask

The clinical score l_c comes from the frozen clinical ensemble in src/models,
which never sees these participants during training. Accelerometry exists only
in this cycle, so the NHANES clinical model, sensor model and fusion layer are
evaluated by repeated nested cross-validation (20 x 5 outer folds, 5 inner
folds), stratified by outcome and availability. In each outer training fold the
two NHANES-fitted branches are fitted, the stacking layer is fitted on their
inner out-of-fold scores, and only then is the held-out fold scored. These are
cross-validated, not external, estimates.

A permutation negative control re-runs the whole procedure with the activity
features shuffled among participants who have a recording: any gain it shows
over the same model without the sensor term is leakage or optimism, not signal.

For deployment every branch and the fusion layer are refitted once on all
participants. Each mode gets the largest threshold reaching sensitivity 0.80 on
its cross-validated predictions: participants with a recording for the fused
score, and all participants scored with m = 0 for clinical-only use.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.prediction.ensemble import (CATEGORICAL_FEATURES, FEATURES,  # noqa: E402
                                     NUMERIC_FEATURES, get_ensemble)
from src.prediction.fusion import (COHORT_MODEL_FILE, FUSION_MODEL_FILE,  # noqa: E402
                                   SENSOR_FEATURES, SENSOR_MODEL_FILE, logit,
                                   stacking_design)

warnings.filterwarnings("ignore")
SEED = 42
N_REPEATS = 20
N_FOLDS = 5
N_INNER_FOLDS = 5
N_NEGATIVE_CONTROL = 50
N_BOOTSTRAP = 1000
TARGET_SENSITIVITY = 0.80
# 246 events for ~15 encoded clinical columns: mild shrinkage keeps the NHANES
# clinical model stable across folds.
COHORT_C = 0.1
# Stacking has six coefficients and ~2,000 rows; a large C leaves it effectively
# unpenalised so the fused probabilities stay calibrated.
STACKING_C = 1e4

DATA = ROOT / "src" / "data" / "PAD_nhanes_2003_2004_sensor.csv"
MODEL_DIR = ROOT / "src" / "models"
RESULTS = ROOT / "src" / "results" / "sensor"
FIGURES = RESULTS / "figures"
for d in (MODEL_DIR, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
y = df["PAD"].to_numpy()
mask = df["Sensor_Available"].to_numpy()
has_recording = mask == 1
X_sensor = df[SENSOR_FEATURES]
print(f"{len(df):,} participants | PAD {y.sum()} ({y.mean():.3f}) | "
      f"valid recordings {has_recording.sum():,} ({y[has_recording].sum()} PAD)")

# Frozen clinical branch: the soft vote of the three deployed pipelines.
ensemble = get_ensemble()
clinical_frame = df[FEATURES].copy()
clinical_frame[NUMERIC_FEATURES] = clinical_frame[NUMERIC_FEATURES].astype("float64")
p_clinical = np.mean([model.predict_proba(clinical_frame)[:, 1]
                      for model in ensemble.models.values()], axis=0)


def cohort_pipeline() -> Pipeline:
    """The clinical features refitted on the real cohort (branch l_r)."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("prep", prep),
                     ("clf", LogisticRegression(C=COHORT_C, max_iter=3000))])


def sensor_pipeline() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=2000))])


def early_fusion_pipeline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC_FEATURES + SENSOR_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES),
                              ("mask", "passthrough", ["Sensor_Available"])])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=1.0, max_iter=3000))])


def stacker() -> LogisticRegression:
    return LogisticRegression(C=STACKING_C, max_iter=2000)


def recalibration_design(clinical_probability, cohort_probability, availability) -> np.ndarray:
    """The fusion design without the m*l_s column: the no-sensor counterpart."""
    return np.delete(stacking_design(clinical_probability, cohort_probability, np.nan,
                                     availability), 2, axis=1)


def inner_oof(make_model, features: pd.DataFrame, outcome: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold branch scores, so the stacker never sees overfitted scores."""
    scores = np.empty(len(outcome))
    folds = StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=seed)
    for fit_idx, score_idx in folds.split(features, outcome):
        model = make_model().fit(features.iloc[fit_idx], outcome[fit_idx])
        scores[score_idx] = model.predict_proba(features.iloc[score_idx])[:, 1]
    return scores


def fit_fusion(train_idx: np.ndarray, sensor_features: pd.DataFrame, seed: int):
    """Fit both NHANES branches, the stacking layer and its no-sensor counterpart."""
    cohort = cohort_pipeline().fit(clinical_frame.iloc[train_idx], y[train_idx])
    p_cohort_train = inner_oof(cohort_pipeline, clinical_frame.iloc[train_idx],
                               y[train_idx], seed)
    with_recording = train_idx[has_recording[train_idx]]
    sensor = sensor_pipeline().fit(sensor_features.iloc[with_recording], y[with_recording])
    p_sensor_train = pd.Series(np.nan, index=train_idx)
    p_sensor_train[with_recording] = inner_oof(
        sensor_pipeline, sensor_features.iloc[with_recording], y[with_recording], seed)
    # Rows with a recording enter a second time with it withheld (m = 0), so the
    # no-recording path learns from every participant, as the app uses it, not
    # only from the minority whose recording failed the wear rules.
    rec = has_recording[train_idx]
    clinical = np.r_[p_clinical[train_idx], p_clinical[train_idx][rec]]
    cohort_scores = np.r_[p_cohort_train, p_cohort_train[rec]]
    sensor_scores = np.r_[p_sensor_train.to_numpy(), np.full(rec.sum(), np.nan)]
    availability = np.r_[mask[train_idx], np.zeros(rec.sum())]
    outcome = np.r_[y[train_idx], y[train_idx][rec]]
    stack = stacker().fit(stacking_design(clinical, cohort_scores, sensor_scores,
                                          availability), outcome)
    recal = stacker().fit(recalibration_design(clinical, cohort_scores, availability), outcome)
    return cohort, sensor, stack, recal


def nested_cv(sensor_features: pd.DataFrame, with_comparators: bool) -> dict:
    """Algorithm 4.3; held-out predictions averaged over the repetitions.

    `clinical_only` scores every held-out participant with m = 0, as the app does
    when no recording is supplied. `recalibrated` is the no-sensor baseline.
    """
    names = ["fused", "sensor", "cohort", "recalibrated", "clinical_only"]
    if with_comparators:
        names.append("early")
    sums = {name: np.zeros(len(df)) for name in names}
    strata = y * 2 + mask
    for repeat in range(N_REPEATS):
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + repeat)
        for train_idx, test_idx in folds.split(df, strata):
            cohort, sensor, stack, recal = fit_fusion(train_idx, sensor_features, SEED + repeat)
            p_cohort_test = cohort.predict_proba(clinical_frame.iloc[test_idx])[:, 1]
            p_sensor_test = np.full(len(test_idx), np.nan)
            held_with = has_recording[test_idx]
            p_sensor_test[held_with] = sensor.predict_proba(
                sensor_features.iloc[test_idx[held_with]])[:, 1]
            sums["cohort"][test_idx] += p_cohort_test
            sums["sensor"][test_idx] += np.nan_to_num(p_sensor_test)
            sums["fused"][test_idx] += stack.predict_proba(stacking_design(
                p_clinical[test_idx], p_cohort_test, p_sensor_test, mask[test_idx]))[:, 1]
            sums["clinical_only"][test_idx] += stack.predict_proba(stacking_design(
                p_clinical[test_idx], p_cohort_test, np.nan, np.zeros(len(test_idx))))[:, 1]
            sums["recalibrated"][test_idx] += recal.predict_proba(recalibration_design(
                p_clinical[test_idx], p_cohort_test, mask[test_idx]))[:, 1]
            if with_comparators:
                early = early_fusion_pipeline().fit(df.iloc[train_idx], y[train_idx])
                sums["early"][test_idx] += early.predict_proba(df.iloc[test_idx])[:, 1]
    predictions = {name: total / N_REPEATS for name, total in sums.items()}
    predictions["sensor"][~has_recording] = np.nan
    return predictions


# ------------------------------------------------------------------ metrics

def calibration(outcome, probability) -> tuple[float, float]:
    """Calibration-in-the-large (slope fixed at 1) and calibration slope."""
    lp = logit(probability)
    intercept = sm.GLM(outcome, np.ones_like(lp), family=sm.families.Binomial(),
                       offset=lp).fit().params[0]
    slope = sm.Logit(outcome, sm.add_constant(lp)).fit(disp=0).params[1]
    return float(intercept), float(slope)


def specificity_at_sensitivity(outcome, probability, target=TARGET_SENSITIVITY) -> float:
    fpr, tpr, _ = roc_curve(outcome, probability)
    return float(1 - fpr[np.argmax(tpr >= target)])


def threshold_at_sensitivity(outcome, probability, target=TARGET_SENSITIVITY) -> float:
    """Largest threshold whose sensitivity reaches the target."""
    _, tpr, thresholds = roc_curve(outcome, probability)
    return float(thresholds[np.argmax(tpr >= target)])


rng = np.random.default_rng(SEED)
BOOT = {}


def bootstrap_indices(rows: np.ndarray) -> list[np.ndarray]:
    """One fixed set of participant resamples per subset, shared across models
    so that AUC differences are paired."""
    key = rows.tobytes()
    if key not in BOOT:
        draws = []
        while len(draws) < N_BOOTSTRAP:
            sample = rng.choice(rows, size=len(rows), replace=True)
            if 0 < y[sample].sum() < len(sample):
                draws.append(sample)
        BOOT[key] = draws
    return BOOT[key]


def auc_interval(rows, probability) -> tuple[float, float]:
    aucs = [roc_auc_score(y[s], probability[s]) for s in bootstrap_indices(rows)]
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def evaluate(model: str, subset: str, rows: np.ndarray, probability: np.ndarray) -> dict:
    outcome, p = y[rows], probability[rows]
    low, high = auc_interval(rows, probability)
    cal_intercept, cal_slope = calibration(outcome, p)
    return {"model": model, "subset": subset, "operating_point": "threshold_free",
            "n": len(rows), "n_pad": int(outcome.sum()),
            "roc_auc": roc_auc_score(outcome, p), "roc_auc_ci_low": low,
            "roc_auc_ci_high": high, "pr_auc": average_precision_score(outcome, p),
            "brier": brier_score_loss(outcome, p), "calibration_intercept": cal_intercept,
            "calibration_slope": cal_slope,
            "specificity_at_sens_0.80": specificity_at_sensitivity(outcome, p)}


def auc_gain(subset: str, rows: np.ndarray, better: np.ndarray, base: np.ndarray,
             model: str, baseline: str) -> dict:
    gains = [roc_auc_score(y[s], better[s]) - roc_auc_score(y[s], base[s])
             for s in bootstrap_indices(rows)]
    low, high = np.percentile(gains, [2.5, 97.5])
    return {"subset": subset, "n": len(rows), "n_pad": int(y[rows].sum()),
            "model": model, "baseline": baseline,
            "auc_baseline": roc_auc_score(y[rows], base[rows]),
            "auc_model": roc_auc_score(y[rows], better[rows]),
            "auc_gain": roc_auc_score(y[rows], better[rows]) - roc_auc_score(y[rows], base[rows]),
            "ci_low": low, "ci_high": high}


def at_threshold(rows: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    outcome, flagged = y[rows], probability[rows] >= threshold
    tp = int(np.sum(flagged & (outcome == 1))); fn = int(np.sum(~flagged & (outcome == 1)))
    fp = int(np.sum(flagged & (outcome == 0))); tn = int(np.sum(~flagged & (outcome == 0)))
    return {"threshold": threshold, "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp), "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn}


# ------------------------------------------------------------------ run

started = time.time()
cv = nested_cv(X_sensor, with_comparators=True)
print(f"nested CV done ({time.time() - started:.0f}s)", flush=True)

ALL = np.arange(len(df))
WITH = np.flatnonzero(has_recording)
WITHOUT = np.flatnonzero(~has_recording)
diabetic = (df["Diabetes"] == "Yes").to_numpy()

MODELS = {"clinical_ensemble": p_clinical, "nhanes_clinical": cv["cohort"],
          "clinical_recalibrated": cv["recalibrated"], "clinical_only": cv["clinical_only"],
          "accelerometry_only": cv["sensor"], "early_fusion": cv["early"],
          "masked_stacking": cv["fused"]}
rows = []
for name, p in MODELS.items():
    rows.append(evaluate(name, "valid_recording", WITH, p))
    if name != "accelerometry_only":
        rows.append(evaluate(name, "all_participants", ALL, p))
for name in ("clinical_recalibrated", "clinical_only", "masked_stacking"):
    rows.append(evaluate(name, "no_recording", WITHOUT, MODELS[name]))

# One operating point per mode the app serves.
deployed_threshold = threshold_at_sensitivity(y[WITH], cv["fused"][WITH])
clinical_threshold = threshold_at_sensitivity(y, cv["clinical_only"])
rows.append({"model": "masked_stacking", "subset": "valid_recording",
             "operating_point": "deployed", "n": len(WITH), "n_pad": int(y[WITH].sum()),
             **at_threshold(WITH, cv["fused"], deployed_threshold)})
rows.append({"model": "clinical_only", "subset": "all_participants",
             "operating_point": "deployed", "n": len(ALL), "n_pad": int(y.sum()),
             **at_threshold(ALL, cv["clinical_only"], clinical_threshold)})
metrics = pd.DataFrame(rows)
metrics.round(4).to_csv(RESULTS / "fusion_metrics.csv", index=False)

# The first four rows are what accelerometry adds: the same stacking with and
# without the sensor term. The last two compare against the frozen ensemble
# alone, which is what the app scored with before the NHANES clinical branch.
gains = pd.DataFrame([
    auc_gain("valid_recording", WITH, cv["fused"], cv["recalibrated"],
             "masked_stacking", "clinical_recalibrated"),
    auc_gain("all_participants", ALL, cv["fused"], cv["recalibrated"],
             "masked_stacking", "clinical_recalibrated"),
    auc_gain("valid_recording_diabetes", WITH[diabetic[WITH]], cv["fused"],
             cv["recalibrated"], "masked_stacking", "clinical_recalibrated"),
    auc_gain("valid_recording_no_diabetes", WITH[~diabetic[WITH]], cv["fused"],
             cv["recalibrated"], "masked_stacking", "clinical_recalibrated"),
    auc_gain("valid_recording", WITH, cv["fused"], p_clinical,
             "masked_stacking", "clinical_ensemble"),
    auc_gain("all_participants", ALL, cv["clinical_only"], p_clinical,
             "clinical_only", "clinical_ensemble"),
])

# Negative control: shuffle activity features among participants with a recording.
observed_gain = gains.loc[0, "auc_gain"]
control = []
for run in range(N_NEGATIVE_CONTROL):
    shuffled = X_sensor.copy()
    order = np.random.default_rng(SEED + 1000 + run).permutation(WITH)
    shuffled.iloc[WITH] = X_sensor.iloc[order].to_numpy()
    shuffled_cv = nested_cv(shuffled, with_comparators=False)
    control.append(roc_auc_score(y[WITH], shuffled_cv["fused"][WITH])
                   - roc_auc_score(y[WITH], shuffled_cv["recalibrated"][WITH]))
    if (run + 1) % 10 == 0:
        print(f"  negative control {run + 1}/{N_NEGATIVE_CONTROL} "
              f"({time.time() - started:.0f}s)", flush=True)
control = np.array(control)
p_permutation = (1 + np.sum(control >= observed_gain)) / (1 + len(control))
gains["permutation_p"] = [p_permutation] + [np.nan] * (len(gains) - 1)
gains.round(4).to_csv(RESULTS / "auc_gain.csv", index=False)
pd.DataFrame({"run": np.arange(1, len(control) + 1), "auc_gain": control}).round(5).to_csv(
    RESULTS / "negative_control.csv", index=False)

# ------------------------------------------------------------------ features

feature_rows = []
for name in SENSOR_FEATURES:
    values = df.loc[has_recording, name]
    pad = values[y[WITH] == 1]; no_pad = values[y[WITH] == 0]
    complete = values.notna().to_numpy()
    auc = roc_auc_score(y[WITH][complete], values[complete])
    feature_rows.append({
        "feature": name, "no_pad_mean": no_pad.mean(), "no_pad_sd": no_pad.std(),
        "pad_mean": pad.mean(), "pad_sd": pad.std(),
        "direction_in_pad": "higher" if auc >= 0.5 else "lower",
        "univariate_auc": max(auc, 1 - auc),
        "corr_with_age": values.corr(df.loc[has_recording, "Age"])})
features = pd.DataFrame(feature_rows)
features.round(4).to_csv(RESULTS / "sensor_feature_summary.csv", index=False)

# ------------------------------------------------------------------ deployment

cohort_model, sensor_model, stacking_model, _ = fit_fusion(ALL, X_sensor, SEED)
joblib.dump(cohort_model, MODEL_DIR / COHORT_MODEL_FILE, compress=3)
joblib.dump(sensor_model, MODEL_DIR / SENSOR_MODEL_FILE, compress=3)
joblib.dump(stacking_model, MODEL_DIR / FUSION_MODEL_FILE, compress=3)

coef = sensor_model.named_steps["clf"].coef_[0]
pd.DataFrame({"feature": SENSOR_FEATURES, "coef_per_sd": coef,
              "odds_ratio_per_sd": np.exp(coef)}).round(4).to_csv(
    RESULTS / "sensor_coefficients.csv", index=False)

b = stacking_model.coef_[0]
pd.DataFrame({"term": ["intercept", "l_c", "l_r", "m*l_s", "m", "m*l_c", "m*l_r"],
              "coefficient": [stacking_model.intercept_[0], *b]}).round(4).to_csv(
    RESULTS / "fusion_coefficients.csv", index=False)

# Relative weight of the branches for patients with a recording, per SD of logit.
deployed_sensor = sensor_model.predict_proba(X_sensor.iloc[WITH])[:, 1]
deployed_cohort = cohort_model.predict_proba(clinical_frame.iloc[WITH])[:, 1]
clinical_weight = (b[0] + b[4]) * logit(p_clinical[WITH]).std()
cohort_weight = (b[1] + b[5]) * logit(deployed_cohort).std()
sensor_weight = b[2] * logit(deployed_sensor).std()

predictions = pd.DataFrame({"SEQN": df["SEQN"], "PAD": y, "Sensor_Available": mask,
                            "p_clinical": p_clinical, "p_nhanes_clinical": cv["cohort"],
                            "p_clinical_recalibrated": cv["recalibrated"],
                            "p_clinical_only": cv["clinical_only"],
                            "p_sensor": cv["sensor"], "p_early_fusion": cv["early"],
                            "p_fused": cv["fused"]})
predictions.round(5).to_csv(RESULTS / "cv_predictions.csv", index=False)

# ------------------------------------------------------------------ figures
C = {"clinical_ensemble": "#2563eb", "clinical_recalibrated": "#0891b2",
     "accelerometry_only": "#059669", "early_fusion": "#d97706", "masked_stacking": "#7c3aed"}
STYLE = {"clinical_ensemble": "-", "clinical_recalibrated": "-.", "accelerometry_only": "--",
         "early_fusion": ":", "masked_stacking": "-"}
LABEL = {"clinical_ensemble": "Clinical ensemble", "clinical_recalibrated": "Clinical branches",
         "accelerometry_only": "Accelerometry only", "early_fusion": "Early fusion (LR)",
         "masked_stacking": "Masked stacking"}
yw = y[WITH]

fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
for ax_, row in zip(axes.flat, feature_rows):
    values = df.loc[has_recording, row["feature"]]
    groups = [values[yw == 0].dropna(), values[yw == 1].dropna()]
    box = ax_.boxplot(groups, tick_labels=["No PAD", "PAD"], widths=0.55,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color="#111", lw=1.5))
    for patch, colour in zip(box["boxes"], ["#94a3b8", "#9e1b2f"]):
        patch.set(facecolor=colour, alpha=0.55, edgecolor="#334155")
    ax_.set_title(f"{row['feature']}\nAUC {row['univariate_auc']:.3f} "
                  f"({row['direction_in_pad']} in PAD)", fontsize=9.5, fontweight="bold")
    ax_.grid(axis="y", alpha=.3)
fig.suptitle(f"Accelerometry features — participants with a valid recording "
             f"(n={len(WITH):,}; {yw.sum()} PAD)", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "01_feature_distributions.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
for m in C:
    p = MODELS[m][WITH]
    fpr, tpr, _ = roc_curve(yw, p)
    ax[0].plot(fpr, tpr, color=C[m], ls=STYLE[m], lw=2.4 if m == "masked_stacking" else 1.8,
               label=f"{LABEL[m]} (AUC={roc_auc_score(yw, p):.3f})")
    pr, rc, _ = precision_recall_curve(yw, p)
    ax[1].plot(rc, pr, color=C[m], ls=STYLE[m], lw=2.4 if m == "masked_stacking" else 1.8,
               label=f"{LABEL[m]} (AP={average_precision_score(yw, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="Chance")
ax[0].set(xlabel="1 − specificity", ylabel="Sensitivity", title="ROC")
ax[1].axhline(yw.mean(), ls="--", c="k", lw=1, alpha=.5, label=f"Prevalence ({yw.mean():.3f})")
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
for a in ax:
    a.legend(fontsize=9); a.grid(alpha=.3); a.title.set_fontweight("bold")
fig.suptitle(f"Sensor branch and fusion — {N_REPEATS}×{N_FOLDS} nested CV, "
             f"participants with valid accelerometry (n={len(WITH):,})",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "02_roc_pr_fusion.png", dpi=160)
plt.close(fig)


def decile_curve(rows_, p):
    bins = pd.qcut(p[rows_], 10, labels=False, duplicates="drop")
    frame = pd.DataFrame({"p": p[rows_], "y": y[rows_], "bin": bins})
    return frame.groupby("bin").mean()


fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
for ax_, (rows_, title) in zip(axes, [(WITH, f"With a recording (m = 1, n={len(WITH):,})"),
                                      (WITHOUT, f"Without a recording (m = 0, n={len(WITHOUT):,})")]):
    for name, colour, label in [("masked_stacking", C["masked_stacking"], "Masked stacking"),
                                ("clinical_ensemble", C["clinical_ensemble"],
                                 "Clinical ensemble (raw, augmented-trained)")]:
        curve = decile_curve(rows_, MODELS[name])
        slope = calibration(y[rows_], MODELS[name][rows_])[1]
        ax_.plot(curve.p, curve.y, "o-", color=colour, lw=1.8, ms=5,
                 label=f"{label}, slope {slope:.2f}")
    top = max(0.45, float(np.percentile(MODELS["clinical_ensemble"][rows_], 95)))
    ax_.plot([0, top], [0, top], "k--", lw=1, alpha=.5, label="Perfect")
    ax_.set(xlabel="Predicted probability (decile mean)", ylabel="Observed PAD fraction",
            title=title, xlim=(0, top), ylim=(0, top))
    ax_.legend(fontsize=9); ax_.grid(alpha=.3); ax_.title.set_fontweight("bold")
fig.suptitle("Calibration by availability pattern — deciles of predicted risk",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "03_calibration_by_mask.png", dpi=160)
plt.close(fig)


def net_benefit(outcome, p, threshold):
    flagged = p >= threshold
    n = len(outcome)
    return (np.sum(flagged & (outcome == 1)) / n
            - np.sum(flagged & (outcome == 0)) / n * threshold / (1 - threshold))


grid = np.linspace(0.01, 0.40, 79)
prevalence = yw.mean()
fig, ax = plt.subplots(figsize=(8, 5.4))
ax.plot(grid, [net_benefit(yw, cv["fused"][WITH], t) for t in grid],
        color=C["masked_stacking"], lw=2.4, label="Masked stacking")
ax.plot(grid, [net_benefit(yw, cv["recalibrated"][WITH], t) for t in grid],
        color=C["clinical_recalibrated"], lw=1.8, label="Clinical branches (no sensor)")
ax.plot(grid, prevalence - (1 - prevalence) * grid / (1 - grid), color="#64748b",
        ls="--", lw=1.2, label="Refer all")
ax.axhline(0, color="#111", lw=1, label="Refer none")
ax.set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-0.01, prevalence + 0.01),
       title=f"Decision curves — participants with valid accelerometry (n={len(WITH):,})")
ax.legend(fontsize=9); ax.grid(alpha=.3); ax.title.set_fontweight("bold")
fig.tight_layout(); fig.savefig(FIGURES / "04_decision_curves.png", dpi=160)
plt.close(fig)

coef_df = pd.read_csv(RESULTS / "sensor_coefficients.csv").sort_values("coef_per_sd")
fig, ax = plt.subplots(figsize=(8.4, 5.4))
ax.barh(coef_df.feature, coef_df.coef_per_sd,
        color=["#9e1b2f" if v > 0 else "#2563eb" for v in coef_df.coef_per_sd])
ax.axvline(0, color="#111", lw=1)
ax.set_xlabel("Standardised log-odds coefficient (per SD; > 0 raises PAD risk)")
ax.set_title("Sensor branch — deployed logistic regression coefficients",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(FIGURES / "05_sensor_coefficients.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
ax[0].hist(control, bins=15, color="#94a3b8", edgecolor="#fff")
ax[0].axvline(observed_gain, color=C["masked_stacking"], lw=2.4,
              label=f"Observed gain {observed_gain:+.4f}")
ax[0].set(xlabel="AUC gain over no-sensor model (shuffled activity features)", ylabel="Runs",
          title=f"Negative control — {N_NEGATIVE_CONTROL} permutations (p = {p_permutation:.3f})")
ax[0].legend(fontsize=9)
order = gains[gains.baseline == "clinical_recalibrated"].iloc[::-1].reset_index(drop=True)
ax[1].errorbar(order.auc_gain, range(len(order)),
               xerr=[order.auc_gain - order.ci_low, order.ci_high - order.auc_gain],
               fmt="o", color=C["masked_stacking"], ecolor="#64748b", capsize=3, ms=7)
ax[1].axvline(0, color="#dc2626", ls="--", lw=1.2)
ax[1].set_yticks(range(len(order)),
                 [f"{s.replace('_', ' ')} (n={n:,})" for s, n in zip(order.subset, order.n)])
ax[1].set(xlabel="AUC gain from adding accelerometry (95% bootstrap CI)",
          title="AUC gain by subset")
for a in ax:
    a.grid(alpha=.3); a.title.set_fontweight("bold")
fig.tight_layout(); fig.savefig(FIGURES / "06_auc_gain_and_negative_control.png", dpi=160)
plt.close(fig)

(RESULTS / "run_metadata.json").write_text(json.dumps({
    "dataset": DATA.name, "n": len(df), "n_pad": int(y.sum()),
    "n_valid_recording": int(has_recording.sum()), "n_pad_valid_recording": int(yw.sum()),
    "sensor_features": SENSOR_FEATURES,
    "clinical_branch": "frozen soft vote of src/models/*_clinical.joblib",
    "nhanes_clinical_branch": f"logistic regression, C={COHORT_C}, fitted on this cohort",
    "protocol": {"repeats": N_REPEATS, "outer_folds": N_FOLDS, "inner_folds": N_INNER_FOLDS,
                 "stratified_by": "PAD x Sensor_Available"},
    "negative_control_runs": N_NEGATIVE_CONTROL, "bootstrap_resamples": N_BOOTSTRAP,
    "deployed_threshold": deployed_threshold,
    "clinical_only_threshold": clinical_threshold,
    "threshold_rule": f"largest score with sensitivity >= {TARGET_SENSITIVITY} on CV "
                      "predictions: participants with a recording (fused), all "
                      "participants scored with m = 0 (clinical only)",
    "branch_weight_per_sd_logit": {"clinical": float(clinical_weight),
                                   "nhanes_clinical": float(cohort_weight),
                                   "sensor": float(sensor_weight)},
    "seed": SEED}, indent=2))

print("\n--- Threshold-free metrics, participants with a recording ---")
print(metrics[(metrics.subset == "valid_recording") & (metrics.operating_point == "threshold_free")]
      [["model", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "brier",
        "calibration_slope", "specificity_at_sens_0.80"]].round(3).to_string(index=False))
print("\n--- AUC gain ---")
print(gains.round(4).to_string(index=False))
print(f"\nNegative control mean gain {control.mean():+.4f} (SD {control.std():.4f}, "
      f"max {control.max():+.4f}); p = {p_permutation:.3f}")
print(f"Deployed thresholds: fused {deployed_threshold:.4f}, "
      f"clinical only {clinical_threshold:.4f}")
print(f"Done ({time.time() - started:.0f}s).")
