"""
classification.py
-----------------
Supervised learning for the high-value customer label.

Target options (cfg.target.type):
    top_quantile : Monetary > given quantile (default 0.75)
    threshold    : Monetary > absolute amount
    top_n        : top-N customers by Monetary

Models: Random Forest, XGBoost (falls back to GradientBoosting if
xgboost is not installed), MLP. Which ones run is set in
cfg.modeling.classifiers.

SMOTE is placed inside an imblearn.pipeline.Pipeline so that it is
fit only on each CV training fold. Applying SMOTE once on the full
training set before cross_val_score leaks information about held-out
positives and pushes CV scores upward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
    from .config import PipelineConfig
    from .utils import RANDOM_SEED, get_logger
except ImportError:
    from config import PipelineConfig  # type: ignore
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
def make_target(rfm: pd.DataFrame, cfg: PipelineConfig) -> pd.Series:
    """Build the binary high-value label per cfg.target.type."""
    t = cfg.target
    m = rfm["Monetary"]
    if t.type == "top_quantile":
        thr = m.quantile(t.quantile)
        LOG.info("High-value: Monetary > %.2f (q=%.2f)", thr, t.quantile)
        return (m > thr).astype(int)
    if t.type == "threshold":
        LOG.info("High-value: Monetary > %.2f (absolute)", t.threshold)
        return (m > t.threshold).astype(int)
    if t.type == "top_n":
        n = min(t.top_n, len(rfm))
        cutoff = m.nlargest(n).min()
        LOG.info("High-value: top %d customers (Monetary >= %.2f)", n, cutoff)
        return (m >= cutoff).astype(int)
    raise ValueError(f"Unknown target type: {t.type}")


def build_features(rfm: pd.DataFrame) -> pd.DataFrame:
    """Return Recency + Frequency. Monetary defines the label so it is dropped."""
    return rfm[["Recency", "Frequency"]].copy()


# ---------------------------------------------------------------------------
# Pipeline pieces
# ---------------------------------------------------------------------------
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float):
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )


def _build_pipeline(model, use_smote: bool) -> ImbPipeline:
    """Build an imblearn Pipeline so scaling and SMOTE happen per CV fold."""
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
    steps.append(("clf", model))
    return ImbPipeline(steps=steps)


def evaluate(name: str, model, X_train, y_train, X_test, y_test,
             cv_folds: int, use_smote: bool) -> ClfResult:
    """Fit the pipeline, compute CV F1, and score on the test set."""
    pipe = _build_pipeline(model, use_smote)

    cv = cross_val_score(
        pipe, X_train, y_train,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True,
                           random_state=RANDOM_SEED),
        scoring="f1", n_jobs=-1,
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cv_f1_mean": float(cv.mean()),
        "cv_f1_std": float(cv.std()),
    }
    fpr, tpr, thr = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True,
                                   zero_division=0)
    LOG.info("%s -> %s", name,
             {k: round(v, 4) for k, v in metrics.items()})
    return ClfResult(name=name, model=pipe, metrics=metrics,
                     y_true=np.asarray(y_test), y_pred=y_pred,
                     y_proba=y_proba, roc_curve=(fpr, tpr, thr),
                     confusion=cm, cv_scores=cv, report=report)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def _make_model(name: str):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                random_state=RANDOM_SEED, eval_metric="logloss",
                tree_method="hist", n_jobs=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            LOG.warning("xgboost not installed -- using GradientBoosting")
            return GradientBoostingClassifier(random_state=RANDOM_SEED)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            max_iter=500, random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown classifier: {name}")


def _display_name(key: str) -> str:
    return {"random_forest": "RandomForest", "xgboost": "XGBoost",
            "mlp": "MLP"}.get(key, key)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_classification(rfm: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """Run the full supervised pipeline driven by cfg.modeling."""
    y = make_target(rfm, cfg)
    X = build_features(rfm)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg.modeling.test_size
    )
    LOG.info("Split: train=%d (pos=%d) / test=%d (pos=%d)",
             len(y_train), int(y_train.sum()),
             len(y_test), int(y_test.sum()))

    results: dict[str, ClfResult] = {}
    for key in cfg.modeling.classifiers:
        name = _display_name(key)
        LOG.info("Training %s ...", name)
        results[name] = evaluate(
            name, _make_model(key),
            X_train, y_train, X_test, y_test,
            cv_folds=cfg.modeling.cv_folds, use_smote=cfg.modeling.smote,
        )

    return {
        "results": results,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": list(X.columns),
    }
