"""Train the educational MediLens AI model on bundled synthetic data.
This is reproducible demo training, NOT clinical model development.
"""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "demo_synthetic_blood_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
df = pd.read_csv(DATA)
features = ["age_years","sex_code_demo","hemoglobin_g_dl","rbc_million_uL","mcv_fL","mch_pg","hematocrit_percent"]
X, y = df[features], df["demo_label_anemia_pattern"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=42))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {
    "dataset_type": "synthetic educational data; not real patient data",
    "random_state": 42, "rows_total": int(len(df)), "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)), "features": features, "target": "demo_label_anemia_pattern",
    "accuracy": round(float(accuracy_score(y_test,pred)),4),
    "precision": round(float(precision_score(y_test,pred,zero_division=0)),4),
    "recall": round(float(recall_score(y_test,pred,zero_division=0)),4),
    "f1": round(float(f1_score(y_test,pred,zero_division=0)),4),
    "confusion_matrix_labels_0_1": confusion_matrix(y_test,pred).tolist(),
    "classification_report": classification_report(y_test,pred,output_dict=True,zero_division=0),
    "warning": "Metrics measure agreement with synthetic labels only; they do not estimate clinical performance."
}
joblib.dump(model, MODEL_DIR / "medilens_model.joblib")
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k:metrics[k] for k in ["rows_total","train_rows","test_rows","accuracy","precision","recall","f1","confusion_matrix_labels_0_1"]}, indent=2))
