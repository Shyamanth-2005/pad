"""
Scenario A — PAD classification from clinical features only (no ABI).

Models: Logistic Regression, Random Forest, XGBoost.
Data:   PAD_dataset_50000.csv (effect sizes fitted to real NHANES odds ratios).

Feature importance is reported four ways, because impurity gain alone is biased
toward high-cardinality continuous features such as Age:
  1. Permutation importance  - drop in test AUC when a feature is shuffled
  2. Mean |SHAP|             - average marginal contribution per prediction
  3. Adjusted odds ratios    - logistic coefficients with 95% CI, clinically readable
  4. Native importance       - impurity gain / |standardised coefficient|, for reference
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42

# Repo layout: this file lives at src/train/, so the project root is two up.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src" / "data" / "PAD_dataset_50000.csv"
MODEL_DIR = ROOT / "src" / "models"
RESULTS = ROOT / "src" / "results"
FIGURES = RESULTS / "figures"
CKPT = ROOT / "src" / "train" / ".cache"
for d in (MODEL_DIR, RESULTS, FIGURES, CKPT):
    d.mkdir(parents=True, exist_ok=True)

# Filenames the API loads from; keep these in sync with src/prediction/ensemble.py.
MODEL_FILES = {
    "LogisticRegression": "logistic_regression_clinical.joblib",
    "RandomForest": "random_forest_clinical.joblib",
    "XGBoost": "xgboost_clinical.joblib",
}

df = pd.read_csv(DATA)
print(f"{len(df):,} rows | PAD prevalence {df.PAD.mean():.4f}")

CAT = ["Sex", "Race", "Smoking", "Diabetes", "Medical_History_Hypertension"]
NUM = ["Age", "BMI", "HbA1c", "SBP", "DBP", "Total_Cholesterol", "HDL", "TC_HDL_Ratio"]
FEATURES = NUM + CAT

y = df["PAD"].values
X = df[FEATURES]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"Train {len(Xtr):,} | Test {len(Xte):,}")


def preproc(scale):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]),
         CAT)])


MODELS = {
    "LogisticRegression": Pipeline([
        ("prep", preproc(True)),
        ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=SEED))]),
    "RandomForest": Pipeline([
        ("prep", preproc(False)),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=12,
                                       min_samples_leaf=12, max_features="sqrt",
                                       n_jobs=1, random_state=SEED))]),
    "XGBoost": Pipeline([
        ("prep", preproc(False)),
        ("clf", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.06,
                              subsample=0.85, colsample_bytree=0.85,
                              min_child_weight=5, reg_lambda=1.5,
                              eval_metric="logloss", n_jobs=1,
                              random_state=SEED, tree_method="hist"))]),
}


def youden(yv, p):
    fpr, tpr, thr = roc_curve(yv, p)
    return float(thr[np.argmax(tpr - fpr)])


def metrics_at(yv, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, yh).ravel()
    return {"threshold": round(thr, 4), "accuracy": accuracy_score(yv, yh),
            "precision_PPV": precision_score(yv, yh, zero_division=0),
            "recall_sensitivity": recall_score(yv, yh),
            "specificity": tn / (tn + fp), "npv": tn / (tn + fn),
            "f1": f1_score(yv, yh), "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn)}


rows, preds, imp_rows, fitted = [], {}, [], {}
for name, pipe in MODELS.items():
    ck = CKPT / f"{name}.pkl"
    if ck.exists():
        st = joblib.load(ck)
        rows += st["rows"]; imp_rows += st["imp"]; preds[name] = st["p"]
        print(f"{name:20s} loaded from checkpoint")
        continue
    _r0, _i0 = len(rows), len(imp_rows)
    cv = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True,
                         random_state=SEED), scoring="roc_auc", n_jobs=1)
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    preds[name] = p
    auc, ap = roc_auc_score(yte, p), average_precision_score(yte, p)
    for lbl, thr in [("default_0.50", 0.50), ("youden", youden(yte, p))]:
        rows.append({"model": name, "operating_point": lbl,
                     "cv_auc_mean": cv.mean(), "cv_auc_std": cv.std(),
                     "test_roc_auc": auc, "test_pr_auc": ap,
                     "brier": brier_score_loss(yte, p), **metrics_at(yte, p, thr)})
    joblib.dump(pipe, MODEL_DIR / MODEL_FILES[name], compress=3)
    print(f"{name:20s} CV-AUC {cv.mean():.4f} | Test AUC {auc:.4f} | PR-AUC {ap:.4f}")

    # ---- 1. permutation importance, grouped back to ORIGINAL columns ----
    pi = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=SEED,
                                scoring="roc_auc", n_jobs=1)
    for f, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std):
        imp_rows.append({"model": name, "feature": f, "method": "permutation_auc_drop",
                         "value": m, "std": s})

    # ---- 2. mean |SHAP|, summed over one-hot columns of each source feature ----
    enc = pipe.named_steps["prep"].get_feature_names_out()
    enc = [e.split("__", 1)[1] for e in enc]
    samp = Xte.sample(min(1500, len(Xte)), random_state=SEED)
    Zs = pipe.named_steps["prep"].transform(samp)
    clf = pipe.named_steps["clf"]
    if name == "LogisticRegression":
        sv = shap.LinearExplainer(clf, Zs).shap_values(Zs)
    else:
        sv = shap.TreeExplainer(clf).shap_values(Zs)
        if isinstance(sv, list):
            sv = sv[1]
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    mabs = np.abs(sv).mean(axis=0)
    agg = {f: 0.0 for f in FEATURES}
    for col, v in zip(enc, mabs):
        src = col if col in agg else next(f for f in CAT if col.startswith(f + "_"))
        agg[src] += v
    for f, v in agg.items():
        imp_rows.append({"model": name, "feature": f, "method": "mean_abs_shap",
                         "value": v, "std": np.nan})

    # ---- 4. native importance ----
    nat = (np.abs(clf.coef_[0]) if name == "LogisticRegression"
           else clf.feature_importances_)
    agg2 = {f: 0.0 for f in FEATURES}
    for col, v in zip(enc, nat):
        src = col if col in agg2 else next(f for f in CAT if col.startswith(f + "_"))
        agg2[src] += v
    for f, v in agg2.items():
        imp_rows.append({"model": name, "feature": f, "method": "native", "value": v,
                         "std": np.nan})
    joblib.dump({"rows": rows[_r0:], "imp": imp_rows[_i0:], "p": p}, ck)
    del pipe, clf, sv, Zs, samp, pi
    import gc; gc.collect()
    print(f"{name:20s} checkpointed", flush=True)

# ---- 3. adjusted odds ratios with 95% CI ----
OR_SPEC = {
    "Age (per 10 y)": df.Age / 10, "Male": (df.Sex == "Male").astype(int),
    "Non-Hispanic Black": (df.Race == "Non-Hispanic Black").astype(int),
    "BMI (per 5)": df.BMI / 5, "Current smoker": (df.Smoking == "Current").astype(int),
    "Former smoker": (df.Smoking == "Former").astype(int),
    "Diabetes": (df.Diabetes == "Yes").astype(int), "HbA1c (per 1%)": df.HbA1c,
    "SBP (per 10)": df.SBP / 10, "DBP (per 10)": df.DBP / 10,
    "Total chol. (per 10)": df.Total_Cholesterol / 10, "HDL (per 10)": df.HDL / 10,
    "Hypertension": (df.Medical_History_Hypertension == "Yes").astype(int)}
Xor = pd.DataFrame(OR_SPEC).astype(float)
msk = Xor.notna().all(1)
ores = sm.Logit(df.PAD.astype(float)[msk], sm.add_constant(Xor[msk])).fit(disp=0)
ci = ores.conf_int()
or_df = pd.DataFrame({"term": ores.params.index, "OR": np.exp(ores.params),
                      "CI_low": np.exp(ci[0]), "CI_high": np.exp(ci[1]),
                      "p_value": ores.pvalues}).drop("const").reset_index(drop=True)
or_df.round(4).to_csv(RESULTS / "odds_ratios.csv", index=False)

res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS / "model_metrics.csv", index=False)
imp = pd.DataFrame(imp_rows)
imp["rank"] = imp.groupby(["model", "method"])["value"].rank(ascending=False)
imp.round(6).sort_values(["method", "model", "rank"]).to_csv(
    RESULTS / "feature_importance.csv", index=False)

pred_df = df.loc[Xte.index, ["Participant_ID", "Age", "Sex", "Smoking",
                             "Diabetes", "ABI_Min", "PAD"]].rename(
    columns={"PAD": "PAD_actual"})
for k, p in preds.items():
    pred_df[f"prob__{k}"] = np.round(p, 5)
pred_df.to_csv(RESULTS / "test_predictions.csv", index=False)

# =============================================================== figures
C = {"LogisticRegression": "#2563eb", "RandomForest": "#059669", "XGBoost": "#d97706"}

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
for m in C:
    fpr, tpr, _ = roc_curve(yte, preds[m])
    ax[0].plot(fpr, tpr, color=C[m], lw=2,
               label=f"{m} (AUC={roc_auc_score(yte, preds[m]):.3f})")
    pr, rc, _ = precision_recall_curve(yte, preds[m])
    ax[1].plot(rc, pr, color=C[m], lw=2,
               label=f"{m} (AP={average_precision_score(yte, preds[m]):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=.6, label="Chance")
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
ax[1].axhline(yte.mean(), ls="--", c="k", lw=1, alpha=.6,
              label=f"Prevalence ({yte.mean():.3f})")
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
for a in ax:
    a.legend(fontsize=9); a.grid(alpha=.3); a.title.set_fontweight("bold")
fig.suptitle("Scenario A — Clinical features only (no ABI)", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "01_roc_pr_curves.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax_, m in zip(axes, C):
    thr = youden(yte, preds[m])
    cm = confusion_matrix(yte, (preds[m] >= thr).astype(int))
    ax_.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum() * 100
            ax_.text(j, i, f"{cm[i,j]:,}\n({pct:.1f}%)", ha="center", va="center",
                     fontsize=11, color="white" if pct > 55 else "#111")
    ax_.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax_.set_yticks([0, 1], ["True 0", "True 1"])
    ax_.set_title(f"{m}\n(threshold {thr:.3f})", fontsize=10, fontweight="bold")
fig.suptitle("Confusion Matrices — Youden-optimal Threshold", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "02_confusion_matrices.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 5.4))
for m in C:
    pt, pp = calibration_curve(yte, preds[m], n_bins=12, strategy="quantile")
    ax.plot(pp, pt, "o-", color=C[m], lw=1.8, ms=4, label=m)
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.6, label="Perfect")
ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive",
       title="Calibration — Scenario A")
ax.legend(); ax.grid(alpha=.3); ax.title.set_fontweight("bold")
fig.tight_layout(); fig.savefig(FIGURES / "03_calibration.png", dpi=160)
plt.close(fig)

# permutation importance — the headline chart
fig, axes = plt.subplots(1, 3, figsize=(15.5, 6))
for ax_, m in zip(axes, C):
    d = imp[(imp.model == m) & (imp.method == "permutation_auc_drop")]
    d = d.sort_values("value")
    ax_.barh(d.feature, d.value, xerr=d["std"], color=C[m],
             error_kw=dict(ecolor="#555", lw=1))
    ax_.set_title(m, fontsize=11, fontweight="bold")
    ax_.set_xlabel("Drop in test ROC-AUC when shuffled")
    ax_.grid(axis="x", alpha=.3)
fig.suptitle("Permutation Importance — unbiased, measured on held-out data",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "04_permutation_importance.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 6))
for ax_, m in zip(axes, C):
    d = imp[(imp.model == m) & (imp.method == "mean_abs_shap")].sort_values("value")
    ax_.barh(d.feature, d.value, color=C[m])
    ax_.set_title(m, fontsize=11, fontweight="bold")
    ax_.set_xlabel("mean |SHAP value|")
    ax_.grid(axis="x", alpha=.3)
fig.suptitle("SHAP Importance — mean absolute contribution per prediction",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "05_shap_importance.png", dpi=160)
plt.close(fig)

# odds ratio forest plot
d = or_df.sort_values("OR")
fig, ax = plt.subplots(figsize=(8.6, 6.4))
ax.errorbar(d.OR, range(len(d)),
            xerr=[d.OR - d.CI_low, d.CI_high - d.OR], fmt="o",
            color="#1e3a8a", ecolor="#64748b", capsize=3, ms=6)
ax.axvline(1, color="#dc2626", ls="--", lw=1.2)
ax.set_yticks(range(len(d)), d.term)
ax.set_xscale("log")
ax.set_xlabel("Adjusted odds ratio (log scale, 95% CI)")
ax.set_title("Adjusted Odds Ratios for PAD — Scenario A features",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(FIGURES / "06_odds_ratios_forest.png", dpi=160)
plt.close(fig)

# method comparison for XGBoost: native vs permutation vs SHAP
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
for ax_, meth, lbl in zip(axes, ["native", "permutation_auc_drop", "mean_abs_shap"],
                          ["Impurity gain (biased)", "Permutation (unbiased)", "mean |SHAP|"]):
    d = imp[(imp.model == "XGBoost") & (imp.method == meth)].sort_values("value")
    d = d.assign(norm=d.value / d.value.sum())
    ax_.barh(d.feature, d.norm, color="#d97706")
    ax_.set_title(lbl, fontsize=11, fontweight="bold")
    ax_.set_xlabel("share of total importance")
    ax_.grid(axis="x", alpha=.3)
fig.suptitle("Why the ranking depends on the method — XGBoost, same model",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "07_importance_method_comparison.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.4, 5)); sub = res[res.operating_point == "youden"]
x = np.arange(3); w = 0.26
for k, (met, lbl) in enumerate([("test_roc_auc", "ROC-AUC"), ("test_pr_auc", "PR-AUC"),
                                ("f1", "F1")]):
    b = ax.bar(x + (k - 1) * w, [sub[sub.model == m][met].iloc[0] for m in C], w, label=lbl)
    ax.bar_label(b, fmt="%.3f", fontsize=8)
ax.set_xticks(x, list(C)); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=.3)
ax.set_title("Scenario A — Model Comparison", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "08_model_comparison.png", dpi=160)
plt.close(fig)

(RESULTS / "run_metadata.json").write_text(json.dumps(
    {"dataset": DATA.name, "n": len(df),
     "prevalence": float(df.PAD.mean()), "features": FEATURES,
     "n_train": len(Xtr), "n_test": len(Xte), "seed": SEED}, indent=2))

print("\n--- Permutation importance (XGBoost, top 8) ---")
print(imp[(imp.model == "XGBoost") & (imp.method == "permutation_auc_drop")]
      .nlargest(8, "value")[["feature", "value"]].to_string(index=False))
print("\nDone.")
