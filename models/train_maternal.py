"""
Train a simple, interpretable logistic regression for maternal risk
(e.g., risk of preeclampsia) using synthetic data.
Saves model to models/maternal_model.pkl
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path

def make_synthetic(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(16, 45, size=n)
    sys_bp = rng.normal(115, 15, size=n).clip(85, 190).astype(int)
    dia_bp = (sys_bp * 0.65 + rng.normal(0, 5, size=n)).clip(50, 120).astype(int)
    hemoglobin = rng.normal(11.8, 1.2, size=n).clip(7.0, 15.0)
    gest_weeks = rng.integers(8, 40, size=n)
    prev_pre = rng.integers(0, 2, size=n)

    # Risk function (synthetic): higher sys_bp, higher age, previous preeclampsia → higher risk
    logit = (
        0.03 * (sys_bp - 120) +
        0.04 * (age - 30) +
        0.7 * prev_pre +
        -0.1 * (hemoglobin - 11.5)
    )
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(size=n) < p).astype(int)

    df = pd.DataFrame({
        "age": age,
        "systolic_bp": sys_bp,
        "diastolic_bp": dia_bp,
        "hemoglobin": hemoglobin,
        "gestational_weeks": gest_weeks,
        "previous_preeclampsia": prev_pre,
        "label": y
    })
    return df

def train_and_save(df: pd.DataFrame, out_path: Path):
    X = df[["age", "systolic_bp", "diastolic_bp", "hemoglobin", "gestational_weeks", "previous_preeclampsia"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, (proba > 0.5).astype(int))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)

    return {"auc": float(auc), "report": report}

if __name__ == "__main__":
    models_dir = Path(__file__).resolve().parent
    out = models_dir / "maternal_model.pkl"
    df = make_synthetic()
    metrics = train_and_save(df, out)
    print("Saved:", out)
    print("AUC:", metrics["auc"])
    print(metrics["report"])
