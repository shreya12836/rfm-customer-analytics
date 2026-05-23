"""
classification.py
-----------------
Supervised learning for high-value customer prediction.

Target definition
~~~~~~~~~~~~~~~~~
A customer is labelled ``high_value = 1`` if their total Monetary value
falls in the **top 25%** of the population (i.e. above the 75th
percentile). All others are labelled 0. This binary framing turns the
business question -- "who should the marketing team prioritise?" --
into a tractable classification problem.

Models
~~~~~~
- Random Forest
- XGBoost (gradient-boosted trees)
- Multi-Layer Perceptron (sklearn MLPClassifier)

Each model is trained on SMOTE-balanced data, scored with 5-fold
stratified cross-validation, and finally evaluated on a held-out test
set with accuracy / precision / recall / F1 / ROC-AUC plus the full
classification report and confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from .utils import RANDOM_SEED, get_logger
except ImportError:
    from utils import RANDOM_SEED, get_logger  # type: ignore

LOG = get_logger("classification")


@dataclass
class ClfResult:
    """Per-model evaluation bundle."""

    name: str
    model: Any
    metrics: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    roc_curve: tuple[np.ndarray, np.ndarray, np.ndarray]
    confusion: np.ndarray
    cv_scores: np.ndarray
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------
def make_target(rfm: pd.DataFrame, quantile: float = 0.75) -> pd.Series:
    """Label customers above the chosen Monetary quantile as high-value."""
    threshold = rfm["Monetary"].quantile(quantile)
    LOG.info("High-value threshold (Monetary > %.2f) at q=%.2f",
             threshold, quantile)
    return (rfm["Monetary"] > threshold).astype(int)


def build_features(rfm: pd.DataFrame) -> pd.DataFrame:
    """Use Recency and Frequency as predictors -- Monetary defines the label
    so it must be excluded to avoid trivial leakage."""
    return rfm[["Recency", "Frequency"]].copy()


# ---------------------------------------------------------------------------
# Pipeline pieces
# ---------------------------------------------------------------------------
def split_and_balance(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25):
    """Stratified split + StandardScaler + SMOTE oversampling on the train set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)
    LOG.info("Train: %d -> %d after SMOTE (positives=%d)",
             len(y_train), len(y_train_bal), int(y_train_bal.sum()))
    return X_train_bal, X_test_s, y_train_bal, y_test, scaler


def evaluate(name: str, model, X_test, y_test, X_train, y_train) -> ClfResult:
    """Fit a model on (X_train, y_train) and score it on (X_test, y_test)."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:                                      # safety fallback
        y_proba = y_pred.astype(float)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    fpr, tpr, thr = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    cv = cross_val_score(
        model, X_train, y_train, cv=StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring="f1",
    )
    metrics["cv_f1_mean"] = float(cv.mean())
    metrics["cv_f1_std"] = float(cv.std())

    report = classification_report(y_test, y_pred, output_dict=True,
                                   zero_division=0)
    LOG.info("%s -> %s", name,
             {k: round(v, 4) for k, v in metrics.items()})
    return ClfResult(name=name, model=model, metrics=metrics,
                     y_true=np.asarray(y_test), y_pred=y_pred,
                     y_proba=y_proba, roc_curve=(fpr, tpr, thr),
                     confusion=cm, cv_scores=cv, report=report)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def get_models() -> dict:
    """Construct the three classifiers used in the study."""
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        random_state=RANDOM_SEED, n_jobs=-1,
    )

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_SEED, eval_metric="logloss",
            tree_method="hist", n_jobs=-1,
        )
    except ImportError:                       # graceful degradation
        from sklearn.ensemble import GradientBoostingClassifier

        LOG.warning("xgboost not installed -- substituting GradientBoosting")
        xgb = GradientBoostingClassifier(random_state=RANDOM_SEED)

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        max_iter=500, random_state=RANDOM_SEED,
    )
    return {"RandomForest": rf, "XGBoost": xgb, "MLP": mlp}


def run_classification(rfm: pd.DataFrame) -> dict:
    """End-to-end supervised learning pipeline."""
    y = make_target(rfm)
    X = build_features(rfm)

    X_train, X_test, y_train, y_test, scaler = split_and_balance(X, y)

    results: dict[str, ClfResult] = {}
    for name, model in get_models().items():
        LOG.info("Training %s ...", name)
        results[name] = evaluate(name, model, X_test, y_test,
                                 X_train, y_train)
    return {
        "results": results,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "feature_names": list(X.columns),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    demo = pd.DataFrame({
        "CustomerID": np.arange(2000),
        "Recency": rng.exponential(30, 2000),
        "Frequency": rng.poisson(5, 2000) + 1,
        "Monetary": rng.lognormal(5, 1.2, 2000),
    })
    out = run_classification(demo)
    for r in out["results"].values():
        print(r.name, r.metrics)
