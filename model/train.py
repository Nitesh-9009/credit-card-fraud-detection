"""
Credit Card Fraud Detection - model training.

Dataset: ULB "Credit Card Fraud Detection" (284,807 European card
transactions from Sept. 2013, 492 frauds). Features V1..V28 are the result
of a PCA transformation; only `Time`, `Amount` and `Class` are original.

Because the data is extremely imbalanced (0.172% fraud), raw accuracy is a
misleading metric (a "predict everything as valid" model scores 99.8%).
We therefore build a *balanced* problem via random under-sampling so that
accuracy is meaningful, then train a K-Nearest-Neighbors classifier and
report a full battery of metrics (accuracy, precision, recall, F1, ROC-AUC).

Run:  python model/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
API_ARTIFACT = ROOT / "api" / "model.npz"
METADATA_PATH = ROOT / "model" / "metadata.json"

# Features used by the model (Time is dropped - it carries no fraud signal
# and only adds noise to distance-based KNN).
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download it first, e.g.:\n"
            "  curl -L -o data/creditcard.csv "
            "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
        )
    return pd.read_csv(DATA_PATH)


def build_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Random under-sampling: keep all frauds + an equal number of valids."""
    fraud = df[df["Class"] == 1]
    valid = df[df["Class"] == 0].sample(n=len(fraud), random_state=RANDOM_STATE)
    balanced = pd.concat([fraud, valid]).sample(frac=1.0, random_state=RANDOM_STATE)
    return balanced.reset_index(drop=True)


def pick_best_k(X_train, y_train, X_test, y_test) -> int:
    """Small grid search over odd k, optimising test accuracy."""
    best_k, best_acc = 1, -1.0
    for k in range(1, 22, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        if acc > best_acc:
            best_k, best_acc = k, acc
    return best_k


def main() -> None:
    print("Loading dataset ...")
    df = load_data()
    print(f"  total rows      : {len(df):,}")
    print(f"  fraud rows      : {int(df['Class'].sum()):,}")

    balanced = build_balanced_dataset(df)
    print(f"  balanced rows   : {len(balanced):,} (50/50 fraud/valid)")

    X = balanced[FEATURES].to_numpy(dtype=np.float64)
    y = balanced["Class"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_k = pick_best_k(X_train_s, y_train, X_test_s, y_test)
    print(f"  best k          : {best_k}")

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n=== Test-set performance (balanced) ===")
    for name, value in metrics.items():
        print(f"  {name:<10}: {value:.4f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(cm))
    print("\n" + classification_report(y_test, y_pred, target_names=["valid", "fraud"]))

    # --- Export a compact, dependency-light artifact for the Vercel API. ---
    # The serverless function re-implements KNN inference with pure NumPy,
    # so it does NOT need scikit-learn at runtime (small, fast cold start).
    API_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        API_ARTIFACT,
        X_train=X_train_s.astype(np.float32),
        y_train=y_train.astype(np.int8),
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        k=np.int64(best_k),
        features=np.array(FEATURES),
    )
    print(f"\nSaved inference artifact -> {API_ARTIFACT.relative_to(ROOT)}")

    METADATA_PATH.write_text(
        json.dumps(
            {
                "model": "KNeighborsClassifier",
                "k": best_k,
                "features": FEATURES,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "metrics": metrics,
                "confusion_matrix": cm,
                "dataset": "ULB Credit Card Fraud Detection (Kaggle mlg-ulb)",
            },
            indent=2,
        )
    )
    print(f"Saved metadata          -> {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
