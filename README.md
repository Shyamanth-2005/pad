# PAD Screening

A web tool that screens a clinical record for peripheral arterial disease. Three
trained models score the record together; a separately measured ankle-brachial
index (ABI) grades severity. When a seven-day hip accelerometer recording is
available, a sensor branch scores it and a fusion layer combines the two scores.

The interface returns three things and nothing else:

| Output | Source |
| --- | --- |
| PAD detected / not detected | Soft-vote ensemble of the three models, fused with the sensor branch when accelerometry is supplied |
| Risk level | The measured ABI, read against the standard bands |
| Confidence | Distance of the ensemble score from the decision threshold |

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
rather than asking for it. `HbA1c`, `Smoking`, `Diabetes` and
`Medical_History_Hypertension` are nullable in the training data and stay
optional in the form — each pipeline imputes them (median for numeric, most
frequent for categorical).

## ABI bands

| ABI | Reading |
| --- | --- |
| 1.00 – 1.40 | Normal blood flow and pressure |
| 0.91 – 0.99 | Borderline, may suggest early changes |
| 0.41 – 0.90 | Mild to moderate PAD |
| 0.00 – 0.40 | Severe PAD, high risk of significant blockages |
| Above 1.40 | Non-compressible — stiff or calcified arteries, often in diabetes |

## Ensemble and threshold

The ensemble probability is the unweighted mean of the three `predict_proba`
outputs. All three models land within 0.002 ROC-AUC of each other, so weighting
them would fit noise.

The detection threshold is the mean of the per-model Youden-optimal thresholds
read at startup from `src/results/model_metrics.csv` — roughly **0.13**, not 0.5.
Youden maximises sensitivity + specificity, which is the right operating point
for screening: at this cutoff the models recall about 89% of true cases, where a
0.5 cutoff would miss nearly half of them. The trade-off is precision of roughly
47%, so a positive result is a prompt to investigate, not a diagnosis.

If `model_metrics.csv` is missing, the threshold falls back to 0.5.

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

The raw ensemble probability is returned separately as `probability`, so nothing
is hidden.

## Sensor branch

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

2. **Sensor model.** A standardised logistic regression maps the ten features to
   `p_s`.

3. **Fusion layer.** Availability-masked logistic stacking, with `m = 1` when a
   valid recording exists:

   ```
   logit p_f = b0 + b1·l_c + b2·m·l_s + b3·m + b4·m·l_c        (l = logit p)
   ```

   With `m = 0` the fused score is a monotone recalibration of the clinical score,
   so missing accelerometry never needs imputing.

**Data.** NHANES 2003–2004 is the only public cohort with both the ABI and
accelerometry. Adults 40+ with an ABI, excluding ABI > 1.40, give 2,455
participants (246 PAD, ABI ≤ 0.90); 1,846 of them (175 PAD) have a valid
recording. The paper reports 1,919. The difference is the calibration flag:
94 participants have `PAXCAL ≠ 1`, and dropping that rule gives 1,924.

**Evaluation.** 20 × 5 repeated nested cross-validation, stratified by outcome
and availability. The sensor model is fitted on training-fold participants with
a recording, and the stacker on inner out-of-fold sensor scores. The clinical
ensemble is the frozen one in `src/models`. Results for participants with a
valid recording:

| Model | ROC-AUC (95% CI) | PR-AUC | Calibration slope |
| --- | --- | --- | --- |
| Clinical ensemble | 0.740 (0.703–0.778) | 0.237 | 0.45 |
| Accelerometry only | 0.751 (0.711–0.788) | 0.251 | 0.93 |
| Early fusion (LR, 23 inputs + mask) | 0.791 (0.755–0.826) | 0.290 | 0.90 |
| **Masked stacking** | **0.781 (0.746–0.817)** | **0.287** | **1.00** |

- **Gain from accelerometry:** +0.041 AUC (0.017–0.063).
- **Negative control:** shuffling the activity features among participants with
  a recording and rerunning everything 50 times gives −0.003 on average
  (max +0.002), so p = 0.02, the smallest attainable with 50 runs.
- **No recording:** the fused calibration slope is 0.97.

The gain is larger than the paper's +0.012 because the clinical branch here is
trained on the augmented 50,000-record cohort. On real NHANES data it ranks
worse (0.740 vs 0.791) and is miscalibrated (slope 0.45), and the stacker
recalibrates it. These are cross-validated estimates, not external validation.

**Threshold.** `0.0694` is the largest fused score that reaches sensitivity 0.80
on the cross-validated predictions (specificity 0.586, NPV 0.966). It is read at
startup from `src/results/sensor/fusion_metrics.csv`. Because it is chosen on the
same predictions, those two figures are not independent estimates.

The sensor branch requires age 40–85, the range in which NHANES measured the
ABI. Consumer wrist wearables are untested: the features describe a hip-worn
research device.

## Layout

```
src/
├─ api/                FastAPI app
│  ├─ main.py          routes, template rendering, static mount
│  └─ schemas.py       request validation, response contract
├─ prediction/
│  ├─ ensemble.py      loads the three pipelines, soft-vote, threshold
│  ├─ fusion.py        sensor model + availability-masked stacking
│  └─ abi.py           ABI band lookup and reference table
├─ templates/
│  └─ index.html       intake form and result readout
├─ static/             style.css, app.js
├─ models/             clinical pipelines, sensor model, stacking layer
├─ data/               PAD_dataset_50000.csv, PAD_nhanes_2003_2004_sensor.csv
├─ results/            clinical metrics, importances, figures
│  └─ sensor/          fusion metrics, AUC gain, negative control, figures
└─ train/
   ├─ train_clinical.py
   ├─ build_sensor_dataset.py   NHANES download, cohort, accelerometer features
   ├─ accelerometer_features.py wear detection and the ten features
   └─ train_sensor.py           nested CV, negative control, deployment refit
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
  "confidence": 0.8618,
  "probability": 0.7598,
  "risk": {
    "value": 0.72,
    "band": "moderate",
    "label": "Mild to moderate PAD",
    "detail": "Reduced flow consistent with peripheral arterial disease."
  },
  "threshold": 0.1309
}
```

The response also carries `modalities` (`["clinical"]`) and
`branch_probabilities` (`{"clinical": 0.7598}`).

Add an `Accelerometry` object to use the sensor branch. The verdict,
`probability`, `confidence` and `threshold` then come from the fused score:

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
`branch_probabilities`, and `"threshold": 0.0694`.

Optional fields accept `null`. Out-of-range values return `422` with the offending
field named, as does accelerometry for a patient under 40.

### `GET /api/health`

Reports which pipelines loaded, the clinical threshold and `sensor_threshold`.

## Retraining

```bash
python src/train/train_clinical.py
```

Reads `src/data/PAD_dataset_50000.csv` and writes the three `.joblib` pipelines to
`src/models/`, metrics and importances to `src/results/`, and figures to
`src/results/figures/`. Per-model checkpoints land in `src/train/.cache/` so an
interrupted run resumes; delete that directory to force a clean retrain. The API
picks up new thresholds on the next restart.

```bash
python src/train/build_sensor_dataset.py   # ~10 min; downloads ~450 MB once
python src/train/train_sensor.py           # ~10 min; mostly the negative control
```

The first command downloads the NHANES 2003–2004 files into `src/data/nhanes_raw/`
(git-ignored) and writes `src/data/PAD_nhanes_2003_2004_sensor.csv`. The second
fits the sensor branch on top of the current clinical pipelines. Run it again
whenever the clinical models change. It writes
`sensor_logistic_regression.joblib` and `fusion_stacking.joblib` to
`src/models/`, and its results and figures to `src/results/sensor/`.

## Tests

```bash
python -m pytest
```

## Scope

Screening support only. This is not a diagnosis, and it does not replace a
vascular assessment.
