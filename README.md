# PAD Screening

A web tool that screens a clinical record for peripheral arterial disease. Two
clinical branches score the record: a three-model ensemble trained on an
augmented 50,000-record cohort, and a logistic regression fitted on real NHANES
patients. A fusion layer combines and calibrates them; a separately measured
ankle-brachial index (ABI) grades severity. When a seven-day hip accelerometer
recording is available, a sensor branch of five models scores it and the fusion
layer adds it in.

The interface returns three things and nothing else:

| Output | Source |
| --- | --- |
| PAD detected / not detected | The fusion layer's score against the threshold for the mode used (clinical only, or clinical + sensor) |
| Risk level | The measured ABI, read against the standard bands |
| Confidence | Distance of the fused score from that threshold |

## Why the ABI is an input, not an output

The models are trained on clinical features only — `ABI_Left`, `ABI_Right` and
`ABI_Min` are excluded along with the `PAD` target. That is the point of the
design: the models estimate PAD risk from data available *without* a vascular
measurement. The ABI you type in is used solely to look up the severity band, so
the verdict and the risk level are two independent readings of the same patient.

## Features the models use

Thirteen columns, in the order the pipelines expect:

**Numeric** — `Age`, `BMI`, `HbA1c`, `SBP`, `DBP`, `Total_Cholesterol`, `HDL`,
`TC_HDL_Ratio`
**Categorical** — `Sex`, `Race`, `Smoking`, `Diabetes`,
`Medical_History_Hypertension`

`TC_HDL_Ratio` is derived from total cholesterol and HDL, so the form computes it
rather than asking for it. `Smoking`, `Diabetes` and
`Medical_History_Hypertension` are required in the form, since a patient can
always answer them. `HbA1c` stays optional there, because it needs a lab test.
All four are nullable in the training data and in the API, and each pipeline
imputes a missing value (median for numeric, most frequent for categorical). A
missing value reaches the pipelines as `NaN`, the value their imputers
recognise. Before, a missing categorical was silently read as its reference
level, so an unknown smoking status scored as "Current".

## ABI bands

| ABI | Reading |
| --- | --- |
| 1.00 – 1.40 | Normal blood flow and pressure |
| 0.91 – 0.99 | Borderline, may suggest early changes |
| 0.41 – 0.90 | Mild to moderate PAD |
| 0.00 – 0.40 | Severe PAD, high risk of significant blockages |
| Above 1.40 | Non-compressible — stiff or calcified arteries, often in diabetes |

## Clinical branches

**Ensemble.** The unweighted mean of the three `predict_proba` outputs. All three
models land within 0.002 ROC-AUC of each other on the augmented cohort (about
0.94), so weighting them would fit noise.

**NHANES clinical model.** On real NHANES patients the ensemble ranks worse
(ROC-AUC 0.733) and is badly overconfident (calibration slope 0.43). A
logistic regression on the same thirteen features, fitted on the real cohort
(`C = 0.1`), reaches 0.762 under the same cross-validation. More flexible models
did not help: gradient boosting scored 0.740 and spline terms 0.761.

Neither branch decides alone. The fusion layer below weighs them, and its
threshold decides.

## Thresholds

Each mode has its own threshold: the largest fused score that still reaches
sensitivity 0.80 on the cross-validated NHANES predictions. Both are read at
startup from `src/results/sensor/fusion_metrics.csv`.

| Mode | Threshold | Sensitivity | Specificity | PPV | NPV |
| --- | --- | --- | --- | --- | --- |
| Clinical only (all 2,455 participants) | 0.0813 | 0.80 | 0.594 | 0.180 | 0.964 |
| Clinical + sensor (1,846 with a recording) | 0.0680 | 0.80 | 0.591 | 0.170 | 0.966 |

At a 10% prevalence a positive result is a prompt to investigate, not a
diagnosis. Each threshold is chosen on the same predictions it is scored on, so
those figures are not independent estimates.

### Confidence

`confidence` measures how far the score sits from the decision boundary, on a
0.5 to 1.0 scale. Reporting the raw probability instead breaks down at a
screening threshold — a positive verdict at 0.14 would read as "PAD detected,
14% confident", which is not what the number means. A score landing exactly on
the threshold gives 0.5; one at either extreme gives 1.0.

```
detected:     0.5 + 0.5 × (p − t) / (1 − t)
not detected: 0.5 + 0.5 × (t − p) / t
```

The fused probability is returned separately as `probability`, and each branch's
own score in `branch_probabilities`, so nothing is hidden.

## Sensor branch and fusion

Implements Sections 3.6–3.8 of the project report. Three parts:

1. **Ten activity features.** These come from a seven-day hip ActiGraph recording
   (minute counts), averaged over valid days. A day is valid when the device was
   worn at least 600 minutes (Troiano non-wear rule) and every minute is flagged
   calibrated and reliable. A recording needs at least four valid days.

   | Volume | Pattern |
   | --- | --- |
   | `Log_Total_Counts` — log(1 + counts/day) | `ASTP` — active→sedentary transition probability |
   | `Sedentary_Share` — wear time < 100 counts/min | `SATP` — sedentary→active transition probability |
   | `Light_Minutes` — 100–2019 counts/min | `Peak30_Counts` — mean of the 30 most active minutes |
   | `MVPA_Minutes` — ≥ 2020 counts/min | `Longest_Active_Bout` — minutes ≥ 100 counts/min |
   | `MVPA_Bout_Minutes` — MVPA in bouts ≥ 10 min | `Log_PM_AM_Ratio` — log of 12–18 h over 6–12 h counts |

2. **Sensor ensemble.** Five models are each trained on the ten features:
   logistic regression, random forest, XGBoost, LightGBM and CatBoost. Their
   probabilities are averaged with non-negative weights that sum to 1 to give
   `p_s` (`src/prediction/sensor.py`). Each fitted model is saved to its own file
   in `src/models/`; at prediction time all five score the patient and their
   decisions are combined with the saved weights. Inside every training fold the
   weights are learned from the five models' out-of-fold validation scores,
   choosing the blend with the lowest log loss, so a model that validates poorly
   gets little say. The settings are fixed in advance rather than tuned, because 175 PAD
   cases are too few to tune on without fitting noise.

   Each model's own test results, on participants with a recording:

   | Model | ROC-AUC (95% CI) | Versus logistic regression | Weight (mean over folds) |
   | --- | --- | --- | --- |
   | Logistic regression | **0.751** (0.711 to 0.788) | reference | 0.68 |
   | Random forest | 0.732 (0.690 to 0.773) | −0.019 | 0.13 |
   | XGBoost | 0.730 (0.686 to 0.770) | −0.021 | 0.16 |
   | CatBoost | 0.721 (0.677 to 0.761) | −0.030 | 0.02 |
   | LightGBM | 0.717 (0.673 to 0.755) | −0.034 | 0.01 |
   | Weighted ensemble | 0.749 (0.709 to 0.786) | −0.002 | n/a |

   On ten smooth activity summaries the tree models find little that a linear
   model misses, and they rank worse. The learned weights reflect that: the
   deployed blend is 0.69 logistic regression and 0.31 random forest, with the
   other three near zero. The ensemble therefore performs about the same as
   logistic regression alone.

   The full testing and validation report is in
   [`src/results/sensor/REPORT.md`](src/results/sensor/REPORT.md). It covers
   per-fold spread and calibration, and sensitivity, specificity, PPV, NPV and
   F1 at two operating points, with CSVs and figures alongside.

3. **Fusion layer.** Availability-masked logistic stacking over the ensemble
   (`l_c`), the NHANES clinical model (`l_r`) and the sensor ensemble (`l_s`), with
   `m = 1` when a valid recording exists:

   ```
   logit p_f = b0 + b1·l_c + b2·l_r + b3·m·l_s + b4·m + b5·m·l_c + b6·m·l_r
   ```

   (`l = logit p`.) Every verdict goes through it. With `m = 0` it combines and
   recalibrates the two clinical branches, so missing accelerometry never needs
   imputing. When it is trained, every participant with a recording also appears
   once with the recording withheld (`m = 0`). The no-recording path therefore
   learns from all 2,455 participants, not just the 609 whose recording failed
   the wear rules. Every branch carries a positive net weight. Per SD of logit,
   for a patient with a recording, the weights are 0.29 for the ensemble, 0.57
   for the NHANES model and 0.59 for the sensor ensemble.

**Data.** NHANES 2003–2004 is the only public cohort with both the ABI and
accelerometry. Adults 40+ with an ABI, excluding ABI > 1.40, give 2,455
participants (246 PAD, ABI ≤ 0.90); 1,846 of them (175 PAD) have a valid
recording. The paper reports 1,919. The difference is the calibration flag:
94 participants have `PAXCAL ≠ 1`, and dropping that rule gives 1,924.

**Evaluation.** 20 × 5 repeated nested cross-validation, stratified by outcome
and availability. Within each outer training fold, the NHANES clinical model and
the five sensor models are fitted, the sensor weights and the stacker are fitted
on their inner out-of-fold scores, and only then is the held-out fold scored.
The clinical ensemble is the frozen one in `src/models`. Outer folds run in
parallel.

Results for participants with a valid recording (n = 1,846, 175 PAD):

| Model | ROC-AUC (95% CI) | PR-AUC | Calibration slope |
| --- | --- | --- | --- |
| Clinical ensemble (raw) | 0.740 (0.703–0.778) | 0.237 | 0.45 |
| NHANES clinical model | 0.777 (0.740–0.812) | 0.261 | 1.15 |
| Clinical branches, no sensor | 0.776 (0.738–0.812) | 0.262 | 1.00 |
| Sensor ensemble (5 models) | 0.749 (0.709–0.786) | 0.256 | 0.94 |
| Early fusion (LR, 23 inputs + mask) | 0.791 (0.755–0.826) | 0.290 | 0.90 |
| **Masked stacking** | **0.789 (0.752–0.826)** | **0.303** | **0.98** |

Clinical-only mode across all 2,455 participants: ROC-AUC **0.762**
(0.730–0.793), calibration slope **1.00**. The raw ensemble scores 0.733 with a
slope of 0.43.

- **Gain from the NHANES branch:** +0.029 AUC (0.014–0.046) in clinical-only
  mode, and +0.049 (0.023–0.072) for the fused score over the raw ensemble.
- **Gain from accelerometry** over the same model without the sensor term:
  +0.013 AUC (−0.003 to 0.028). It was +0.041 before the NHANES branch existed,
  because much of that gain was the stacker correcting a poor clinical score; a
  better clinical branch leaves less for the sensor to add.
- **Negative control:** shuffling the activity features among participants with
  a recording and rerunning everything 50 times gives −0.001 on average
  (max +0.004), so p = 0.02, the smallest attainable with 50 runs. The sensor
  signal is real, but its 95% interval now touches zero. Each control run uses
  5 × 5 folds to keep the run time down; its gain is a difference within one
  run, so the repeat count cancels.

These are cross-validated estimates, not external validation.

**In the form.** One *Run screening* button scores from any tab. The clinical
record is always required. The Sensor tab starts blank: left blank, the record
is scored in clinical-only mode; with all ten fields filled, the sensor branch
is added. A partly filled Sensor tab is rejected rather than imputed.

The Sensor tab asks for everyday units and converts them before calling the
API. Counts per day becomes `Log_Total_Counts = ln(1 + counts)`; the three
percentages become 0–1 shares (`Sedentary_Share`, `ASTP`, `SATP`); and the
afternoon-to-morning ratio becomes `Log_PM_AM_Ratio = ln(ratio)`. The model
averages the daily logs, so taking the log of a weekly average is a close
approximation, not an exact match. Each field's hint gives the NHANES
interquartile range as a typical value.

The form accepts ages 18–85. NHANES measured the ABI only from age 40, so for
ages 18–39 the NHANES clinical model, the sensor ensemble and the fusion layer are
all extrapolating, with no validation data there. Consumer wrist wearables are
also untested: the features describe a hip-worn research device.

## Examples

Every result below comes from the current models through `POST /api/predict`.
Typing the same values into the form gives the same verdict. The verdict
compares the combined probability with the threshold for the mode used: 8.13%
for clinical only, 6.80% with sensor data. The risk level is read from the ABI
alone, so it can disagree with the verdict.

### Clinical only

**PAD detected.** A 71-year-old man who smokes, with diabetes and hypertension:

```json
{
  "Age": 71, "Sex": "Male", "Race": "Non-Hispanic Black", "BMI": 31.2,
  "Smoking": "Current", "Diabetes": "Yes", "HbA1c": 8.4,
  "SBP": 158, "DBP": 88, "Total_Cholesterol": 236, "HDL": 38,
  "Medical_History_Hypertension": "Yes", "ABI": 0.72
}
```

**No PAD detected.** A 45-year-old woman who has never smoked, with normal
blood pressure and lipids:

```json
{
  "Age": 45, "Sex": "Female", "Race": "Non-Hispanic White", "BMI": 24.0,
  "Smoking": "Never", "Diabetes": "No", "HbA1c": 5.2,
  "SBP": 112, "DBP": 70, "Total_Cholesterol": 180, "HDL": 65,
  "Medical_History_Hypertension": "No", "ABI": 1.10
}
```

| Record | Combined probability | Verdict | Confidence | Risk level |
| --- | --- | --- | --- | --- |
| 71-year-old man | 42.7% | PAD detected | 68.8% | Mild to moderate PAD |
| 45-year-old woman | 1.2% | No PAD detected | 92.6% | Normal |

### Clinical + sensor

Sensor data matters most for borderline records, so these two examples are
records the clinical data alone scores close to the threshold. Each uses one of
two activity profiles. The Sensor tab asks for everyday units, which the page
converts before sending:

| Sensor tab field | Inactive | Active |
| --- | --- | --- |
| Daily movement score | 73100 | 211000 |
| Time spent still (%) | 76 | 58.5 |
| Light activity (min/day) | 190 | 331 |
| Exercise (min/day) | 0.5 | 10.9 |
| Exercise without a break (min/day) | 0 | 0 |
| Stopping after moving (per 100 min) | 42 | 27.5 |
| Starting after sitting (per 100 min) | 12 | 19.4 |
| Most active half hour | 700 | 1928 |
| Longest active stretch (min) | 12 | 28.8 |
| Afternoon vs morning (ratio) | 1.11 | 1.28 |

The same profiles as the API's `Accelerometry` object, in model units:

```jsonc
// Inactive: low, fragmented activity, as seen with PAD
{
  "Log_Total_Counts": 11.2, "Sedentary_Share": 0.76, "Light_Minutes": 190,
  "MVPA_Minutes": 0.5, "MVPA_Bout_Minutes": 0, "ASTP": 0.42, "SATP": 0.12,
  "Peak30_Counts": 700, "Longest_Active_Bout": 12, "Log_PM_AM_Ratio": 0.1
}

// Active: the medians of NHANES participants without PAD
{
  "Log_Total_Counts": 12.26, "Sedentary_Share": 0.585, "Light_Minutes": 331,
  "MVPA_Minutes": 10.9, "MVPA_Bout_Minutes": 0, "ASTP": 0.275, "SATP": 0.194,
  "Peak30_Counts": 1928, "Longest_Active_Bout": 28.8, "Log_PM_AM_Ratio": 0.25
}
```

**PAD detected.** A 60-year-old man who has never smoked, with hypertension,
plus the *Inactive* profile:

```json
{
  "Age": 60, "Sex": "Male", "Race": "Non-Hispanic White", "BMI": 28.0,
  "Smoking": "Never", "Diabetes": "No", "HbA1c": 5.8,
  "SBP": 136, "DBP": 78, "Total_Cholesterol": 205, "HDL": 48,
  "Medical_History_Hypertension": "Yes", "ABI": 0.88
}
```

**No PAD detected.** The same man at 64, a former smoker, with an ABI of 1.05
(`"Age": 64, "Smoking": "Former", "ABI": 1.05`), plus the *Active* profile.

| Record | Clinical only | With sensor data | Confidence | Risk level |
| --- | --- | --- | --- | --- |
| 60, never smoked + Inactive | 4.7%, No PAD detected | **11.6%, PAD detected** | 52.6% | Mild to moderate PAD |
| 64, former smoker + Active | 8.8%, PAD detected | **6.5%, No PAD detected** | 52.0% | Normal |

In both, the activity data reverses the clinical-only verdict. Confidence stays
near 50% because the scores sit close to the threshold, which is what that number
is meant to show. For clear-cut records the sensor data confirms rather than
reverses: the 71-year-old with the *Inactive* profile scores 55.0% (PAD
detected), and the 45-year-old with the *Active* profile scores 1.3% (no PAD).

The 60-year-old also shows why the ABI is graded separately. On clinical data
alone the screen misses him, but his measured ABI of 0.88 still reads as mild to
moderate PAD.

## Layout

```
src/
├─ api/                FastAPI app
│  ├─ main.py          routes, template rendering, static mount
│  └─ schemas.py       request validation, response contract
├─ prediction/
│  ├─ ensemble.py      loads the three pipelines, soft-vote
│  ├─ fusion.py        NHANES clinical model, masked stacking, thresholds
│  ├─ sensor.py        loads the five sensor models, combines them by weight
│  └─ abi.py           ABI band lookup and reference table
├─ templates/
│  └─ index.html       intake form and result readout
├─ static/             style.css, app.js
├─ models/             clinical pipelines, NHANES clinical model, five sensor models + weights, stacking layer
├─ data/               PAD_dataset_50000.csv, PAD_nhanes_2003_2004_sensor.csv
├─ results/            clinical metrics, importances, figures
│  └─ sensor/          REPORT.md, per-model comparison, fusion metrics, figures
└─ train/
   ├─ train_clinical.py
   ├─ build_sensor_dataset.py   NHANES download, cohort, accelerometer features
   ├─ accelerometer_features.py wear detection and the ten features
   └─ train_sensor.py           nested CV, per-model reports, negative control, refit
tests/                 feature extraction, fusion and API tests
run.py                 development entrypoint
```

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

python run.py
```

`scikit-learn` is pinned to 1.8.0 because that is the version the `.joblib`
pipelines were fitted with. Loading them under an older release fails —
`LogisticRegression` lost its `multi_class` attribute in 1.7, so 1.6 cannot
unpickle a 1.8 model.

Open <http://127.0.0.1:8000>. Interactive API docs are at `/docs`.

## API

### `POST /api/predict`

```json
{
  "Age": 71, "Sex": "Male", "Race": "Non-Hispanic Black", "BMI": 31.2,
  "Smoking": "Current", "Diabetes": "Yes", "HbA1c": 8.4,
  "SBP": 158, "DBP": 88, "Total_Cholesterol": 236, "HDL": 38,
  "Medical_History_Hypertension": "Yes",
  "ABI": 0.72
}
```

```json
{
  "pad_detected": true,
  "verdict": "PAD detected",
  "confidence": 0.6882,
  "probability": 0.4271,
  "risk": {
    "value": 0.72,
    "band": "moderate",
    "label": "Mild to moderate PAD",
    "detail": "Reduced flow consistent with peripheral arterial disease."
  },
  "threshold": 0.0813,
  "modalities": ["clinical"],
  "branch_probabilities": {"clinical": 0.9798, "nhanes": 0.3601}
}
```

The ensemble alone would put this patient at 98%; on real patients the
calibrated score is 43%. Both are well above the threshold.

Add an `Accelerometry` object to use the sensor branch. The verdict,
`probability`, `confidence` and `threshold` then come from the sensor-mode
score:

```json
{
  "...clinical fields...": "as above",
  "Accelerometry": {
    "Log_Total_Counts": 11.2, "Sedentary_Share": 0.76, "Light_Minutes": 190,
    "MVPA_Minutes": 0.5, "MVPA_Bout_Minutes": 0, "ASTP": 0.42, "SATP": 0.12,
    "Peak30_Counts": 700, "Longest_Active_Bout": 12, "Log_PM_AM_Ratio": 0.1
  }
}
```

The response then has `"modalities": ["clinical", "sensor"]`, a `sensor` entry in
`branch_probabilities` (0.342 here), `"probability": 0.5495` and
`"threshold": 0.068`.

Optional fields accept `null`. Out-of-range values return `422` with the offending
field named.

### `GET /api/health`

Reports which ensemble pipelines loaded, the clinical-only `threshold` and
`sensor_threshold`.

## Retraining

```bash
python src/train/train_clinical.py
```

Reads `src/data/PAD_dataset_50000.csv` and writes the three `.joblib` pipelines to
`src/models/`, metrics and importances to `src/results/`, and figures to
`src/results/figures/`. Per-model checkpoints land in `src/train/.cache/` so an
interrupted run resumes; delete that directory to force a clean retrain.

```bash
python src/train/build_sensor_dataset.py   # ~10 min; downloads ~450 MB once
python src/train/train_sensor.py           # ~1 h on 16 cores; mostly the negative control
```

The first command downloads the NHANES 2003–2004 files into `src/data/nhanes_raw/`
(git-ignored) and writes `src/data/PAD_nhanes_2003_2004_sensor.csv`. The second
fits the NHANES clinical model, the sensor branch and the fusion layer on top of
the current ensemble, and sets both thresholds. Run it again whenever the
clinical pipelines change. It writes `nhanes_clinical_logistic_regression.joblib`,
one file per sensor model (`sensor_logistic_regression.joblib`,
`sensor_random_forest.joblib`, `sensor_xgboost.joblib`, `sensor_lightgbm.joblib`,
`sensor_catboost.joblib`), their weights in `sensor_ensemble_weights.json`, and
`fusion_stacking.joblib` to `src/models/`, and its
results, figures and `REPORT.md` to `src/results/sensor/`. Outer folds run in
parallel on every core (`N_JOBS` in the script). The API picks up new models and
thresholds on the next restart.

## Tests

```bash
python -m pytest
```

## Scope

Screening support only. This is not a diagnosis, and it does not replace a
vascular assessment.
