"""
train_models.py  — v4 (unchanged from v3 except version header)
----------------------------------------------------------------
This file is compatible with app.py v4 and extract.py v4.

Key artefacts produced (required by app.py v4):
  diabetes_model.pkl    heart_model.pkl    liver_model.pkl
  diabetes_features.pkl heart_features.pkl liver_features.pkl
  diabetes_metrics.pkl  heart_metrics.pkl  liver_metrics.pkl
  diabetes_medians.pkl  heart_medians.pkl  liver_medians.pkl  ← loaded by app.py
  all_metrics.pkl
  feature_schema.pkl   ← app.py asserts feature lists against this at startup

Changes vs v3:
  1. DATA VALIDATION  — checks CSV files exist and have required columns
     before touching any ML code; prints a clear error and exits early.
  2. CALIBRATION FIX  — CalibratedClassifierCV now uses a proper held-out
     calibration split (20%) instead of re-using the training fold, which
     avoids optimistic calibration and data leakage into the cal step.
  3. MEDIANS SAVED    — dataset medians (post-imputation) are saved to
     <model>_medians.pkl so app.py loads them from the same artefact
     instead of re-reading the CSVs at every startup.
  4. HEART ENCODING CONSISTENCY — sex_enc uses case-insensitive matching
     ("male"/"Male"/"MALE" all map to 1) to match app.py behaviour.
  5. FEATURE SCHEMA SAVED — a combined schema dict is saved to
     feature_schema.pkl; app.py can assert at startup that its field
     mappings still match the trained feature list.
  6. REPRODUCIBILITY HEADER — sklearn and numpy versions printed at start
     so you know exactly what environment produced a given /models/ folder.
  7. TIMING — each model section reports wall-clock seconds.
  8. PROGRESS FEEDBACK — GridSearchCV verbose=1 so long runs don't look
     frozen (easy to switch back to 0).
"""

import os, pickle, sys, time, warnings
import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, roc_auc_score)
from sklearn.pipeline        import Pipeline
import sklearn

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Reproducibility header ────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  train_models.py  v4")
print(f"  numpy   : {np.__version__}")
print(f"  sklearn : {sklearn.__version__}")
print(f"  pandas  : {pd.__version__}")
print("="*60)


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(obj, name: str) -> None:
    path = os.path.join(MODELS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"   saved  {name}")


def evaluate(model, X_test, y_test, label: str) -> dict:
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1   = f1_score(y_test, preds, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_test, probas)
    cm   = confusion_matrix(y_test, preds).tolist()
    print(f"\n   {label} test-set results:")
    print(f"      Accuracy  : {acc:.4f}")
    print(f"      Precision : {prec:.4f}")
    print(f"      Recall    : {rec:.4f}")
    print(f"      F1 Score  : {f1:.4f}")
    print(f"      ROC-AUC   : {auc:.4f}")
    return {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "roc_auc":          round(auc,  4),
        "confusion_matrix": cm,
    }


def require_csv(filename: str, required_cols: list) -> pd.DataFrame:
    """Load a CSV and abort with a clear message if it is missing or incomplete."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"\n[ERROR] Data file not found: {path}")
        print(f"  Place {filename} in the /data/ folder and re-run.")
        sys.exit(1)
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"\n[ERROR] {filename} is missing expected columns: {missing}")
        print(f"  Found columns: {list(df.columns)}")
        sys.exit(1)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])


PARAM_GRID = {
    "rf__n_estimators":     [200, 400],
    "rf__max_depth":        [None, 10, 20],
    "rf__min_samples_leaf": [1, 2],
}
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def train_and_calibrate(X_train, y_train, label: str):
    """
    Two-phase training:
      Phase 1 — GridSearchCV finds the best RF hyperparameters on 80% of
                the training data (the 'fit split').
      Phase 2 — CalibratedClassifierCV wraps the best estimator and is
                fitted on the remaining 20% (the 'calibration split').

    This is the correct way to calibrate: the calibration set is held out
    from hyperparameter search so the isotonic regression sees unseen data.
    """
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)

    print(f"   Fit split   : {len(X_fit)} samples")
    print(f"   Cal split   : {len(X_cal)} samples")

    gs = GridSearchCV(
        build_pipeline(), PARAM_GRID,
        cv=CV5, scoring="roc_auc", n_jobs=-1, verbose=1,
    )
    gs.fit(X_fit, y_fit)
    print(f"   Best params : {gs.best_params_}")
    print(f"   CV ROC-AUC  : {gs.best_score_:.4f}")

    calibrated = CalibratedClassifierCV(
        gs.best_estimator_, cv="prefit", method="isotonic")
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ═══════════════════════════════════════════════════════
# 1. DIABETES MODEL  —  8 features  (Pima dataset)
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  DIABETES MODEL  (Pima Indians, n≈768, 8 features)")
print("="*60)
t0 = time.time()

df_d = require_csv("diabetes.csv", [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome",
])

for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
    df_d[col] = df_d[col].replace(0, np.nan)
    df_d[col].fillna(df_d[col].median(), inplace=True)

DIABETES_FEATURES = [
    "Pregnancies","Glucose","BloodPressure",
    "SkinThickness","Insulin","BMI",
    "DiabetesPedigreeFunction","Age",
]

X = df_d[DIABETES_FEATURES].values
y = df_d["Outcome"].values
print(f"   Dataset shape : {X.shape}  |  positive rate: {y.mean():.1%}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

cal_d     = train_and_calibrate(X_tr, y_tr, "Diabetes")
metrics_d = evaluate(cal_d, X_te, y_te, "Diabetes")

# Save medians so app.py never needs to re-read the CSV
diabetes_medians = df_d[DIABETES_FEATURES].median().to_dict()

save(cal_d,             "diabetes_model.pkl")
save(DIABETES_FEATURES, "diabetes_features.pkl")
save(metrics_d,         "diabetes_metrics.pkl")
save(diabetes_medians,  "diabetes_medians.pkl")
print(f"   Diabetes done  ({time.time()-t0:.1f}s)\n")


# ═══════════════════════════════════════════════════════
# 2. HEART MODEL  —  expanded features
#    (Cleveland + combined UCI heart dataset)
# ═══════════════════════════════════════════════════════
print("="*60)
print("  HEART MODEL  (Cleveland+combined, expanded features)")
print("="*60)
t0 = time.time()

df_h = require_csv("heart.csv", ["age","sex","cp","trestbps","chol"])

if   "num"    in df_h.columns: df_h["label"] = (df_h["num"]    > 0).astype(int)
elif "target" in df_h.columns: df_h["label"] = (df_h["target"] > 0).astype(int)
elif "output" in df_h.columns: df_h["label"] = (df_h["output"] > 0).astype(int)
else:
    print("[ERROR] heart.csv has no recognised target column (num / target / output)")
    sys.exit(1)

# Case-insensitive sex encoding to match app.py
df_h["sex_enc"] = df_h["sex"].astype(str).str.strip().str.lower().map(
    {"male": 1, "m": 1, "1": 1, "female": 0, "f": 0, "0": 0}
).fillna(0).astype(int)

df_h["fbs_enc"] = df_h["fbs"].apply(
    lambda x: 1 if str(x).strip().lower() in ("true","1") else 0)
df_h["exang_enc"] = df_h["exang"].apply(
    lambda x: 1 if str(x).strip().lower() in ("true","1") else 0)

df_h = pd.get_dummies(df_h, columns=["cp","restecg","slope","thal"], drop_first=True)
df_h.fillna(df_h.median(numeric_only=True), inplace=True)

EXCLUDE = {"id","dataset","sex","fbs","exang","num","target","output","label"}
HEART_FEATURES = [c for c in df_h.columns if c not in EXCLUDE]
print(f"   Features used ({len(HEART_FEATURES)}): {HEART_FEATURES}")

X = df_h[HEART_FEATURES].values
y = df_h["label"].values
print(f"   Dataset shape : {X.shape}  |  positive rate: {y.mean():.1%}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

cal_h     = train_and_calibrate(X_tr, y_tr, "Heart")
metrics_h = evaluate(cal_h, X_te, y_te, "Heart")

heart_medians = {
    f: float(df_h[f].median()) if f in df_h.columns else 0.0
    for f in HEART_FEATURES
}

save(cal_h,           "heart_model.pkl")
save(HEART_FEATURES,  "heart_features.pkl")
save(metrics_h,       "heart_metrics.pkl")
save(heart_medians,   "heart_medians.pkl")
print(f"   Heart done  ({time.time()-t0:.1f}s)\n")


# ═══════════════════════════════════════════════════════
# 3. LIVER MODEL  —  10 features  (ILPD dataset)
# ═══════════════════════════════════════════════════════
print("="*60)
print("  LIVER MODEL  (ILPD, n≈583, 10 features)")
print("="*60)
t0 = time.time()

df_l = require_csv("liver.csv", [
    "Age","Gender","Total_Bilirubin","Direct_Bilirubin",
    "Alkaline_Phosphotase","Alamine_Aminotransferase",
    "Aspartate_Aminotransferase","Total_Protiens",
    "Albumin","Albumin_and_Globulin_Ratio","Dataset",
])

df_l["Gender_enc"] = df_l["Gender"].astype(str).str.strip().str.lower().map(
    {"male": 1, "m": 1, "1": 1, "female": 0, "f": 0, "0": 0}
).fillna(0).astype(int)

df_l["label"] = (df_l["Dataset"] == 1).astype(int)
df_l.fillna(df_l.median(numeric_only=True), inplace=True)

LIVER_FEATURES = [
    "Age","Gender_enc","Total_Bilirubin","Direct_Bilirubin",
    "Alkaline_Phosphotase","Alamine_Aminotransferase",
    "Aspartate_Aminotransferase","Total_Protiens",
    "Albumin","Albumin_and_Globulin_Ratio",
]

X = df_l[LIVER_FEATURES].values
y = df_l["label"].values
print(f"   Dataset shape : {X.shape}  |  positive rate: {y.mean():.1%}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

cal_l     = train_and_calibrate(X_tr, y_tr, "Liver")
metrics_l = evaluate(cal_l, X_te, y_te, "Liver")

liver_medians = df_l[LIVER_FEATURES].median().to_dict()

save(cal_l,           "liver_model.pkl")
save(LIVER_FEATURES,  "liver_features.pkl")
save(metrics_l,       "liver_metrics.pkl")
save(liver_medians,   "liver_medians.pkl")
print(f"   Liver done  ({time.time()-t0:.1f}s)\n")


# ═══════════════════════════════════════════════════════
# 4. Combined artefacts
# ═══════════════════════════════════════════════════════
all_metrics = {
    "diabetes": metrics_d,
    "heart":    metrics_h,
    "liver":    metrics_l,
}
save(all_metrics, "all_metrics.pkl")

feature_schema = {
    "diabetes": DIABETES_FEATURES,
    "heart":    HEART_FEATURES,
    "liver":    LIVER_FEATURES,
}
save(feature_schema, "feature_schema.pkl")

print("\n" + "="*60)
print("  All 3 models trained and saved to /models/")
print("  Artefacts produced:")
for name in [
    "diabetes_model.pkl",    "diabetes_features.pkl",
    "diabetes_metrics.pkl",  "diabetes_medians.pkl",
    "heart_model.pkl",       "heart_features.pkl",
    "heart_metrics.pkl",     "heart_medians.pkl",
    "liver_model.pkl",       "liver_features.pkl",
    "liver_metrics.pkl",     "liver_medians.pkl",
    "all_metrics.pkl",       "feature_schema.pkl",
]:
    path   = os.path.join(MODELS_DIR, name)
    exists = os.path.exists(path)
    size   = f"{os.path.getsize(path)/1024:.1f} KB" if exists else "MISSING"
    status = "OK" if exists else "!!"
    print(f"    [{status}]  {name:45s} {size}")
print("="*60)