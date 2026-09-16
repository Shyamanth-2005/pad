# PAD Screening

A web tool that screens a clinical record for peripheral arterial disease. Three
trained models score the record together; a separately measured ankle-brachial
index (ABI) grades severity.

The interface returns three things and nothing else:

| Output | Source |
| --- | --- |
| PAD detected / not detected | Soft-vote ensemble of the three models |
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

## Layout

```
src/
├─ api/                FastAPI app
│  ├─ main.py          routes, template rendering, static mount
│  └─ schemas.py       request validation, response contract
├─ prediction/
│  ├─ ensemble.py      loads the three pipelines, soft-vote, threshold
│  └─ abi.py           ABI band lookup and reference table
├─ templates/
│  └─ index.html       intake form and result readout
├─ static/             style.css, app.js
├─ models/             three trained .joblib pipelines
├─ data/               PAD_dataset_50000.csv
├─ results/            metrics, importances, figures
└─ train/
   └─ train_clinical.py
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

Optional fields accept `null`. Out-of-range values return `422` with the offending
field named.

### `GET /api/health`

Reports which pipelines loaded and the active threshold.

## Retraining

```bash
python src/train/train_clinical.py
```

Reads `src/data/PAD_dataset_50000.csv` and writes the three `.joblib` pipelines to
`src/models/`, metrics and importances to `src/results/`, and figures to
`src/results/figures/`. Per-model checkpoints land in `src/train/.cache/` so an
interrupted run resumes; delete that directory to force a clean retrain. The API
picks up new thresholds on the next restart.

## Scope

Screening support only. This is not a diagnosis, and it does not replace a
vascular assessment.
