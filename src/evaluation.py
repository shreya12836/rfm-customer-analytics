"""
evaluation.py
-------------
Aggregation of clustering & classification metrics into publication-ready
CSV tables, plus SHAP-based explainability for the tree models.

CSV outputs (under outputs/tables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- clustering_metrics.csv      : silhouette / DB / CH per algorithm
- cluster_summary.csv         : mean RFM per cluster (KMeans)
- classification_metrics.csv  : accuracy / precision / recall / F1 / AUC
- classification_report_*.csv : sklearn classification_report per model
- model_comparison.csv        : final ranking by F1 + ROC-AUC
- shap_feature_importance.csv : mean |SHAP| per feature
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .utils import get_logger, get_paths
except ImportError:
    from utils import get_logger, get_paths  # type: ignore

LOG = get_logger("evaluation")


# ---------------------------------------------------------------------------
# Clustering tables
# ---------------------------------------------------------------------------
def clustering_metrics_table(results: dict) -> pd.DataFrame:
    """Convert clustering results dict -> tidy metrics dataframe."""
    rows = []
    for key, res in results.items():
        if key.startswith("_"):
            continue
        row = {"algorithm": res.name, **res.metrics}
        rows.append(row)
    df = pd.DataFrame(rows)
    LOG.info("Clustering metrics table:\n%s", df.to_string(index=False))
    return df


def cluster_summary_table(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean RFM and customer count per cluster (KMeans labels)."""
    df = rfm.copy()
    df["Cluster"] = labels
    summary = (
        df.groupby("Cluster")
        .agg(Customers=("CustomerID", "count"),
             Recency=("Recency", "mean"),
             Frequency=("Frequency", "mean"),
             Monetary=("Monetary", "mean"))
        .round(2)
        .reset_index()
    )
    # Business-style label
    rank = summary["Monetary"].rank(ascending=False).astype(int)
    name_map = {1: "Champions", 2: "Loyal", 3: "Potential",
                4: "At-Risk", 5: "One-Time", 6: "Hibernating"}
    summary["Segment"] = rank.map(lambda r: name_map.get(r, f"Cluster_{r}"))
    LOG.info("Cluster summary:\n%s", summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Classification tables
# ---------------------------------------------------------------------------
def classification_metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({"model": name, **res.metrics})
    df = pd.DataFrame(rows).round(4)
    LOG.info("Classification metrics:\n%s", df.to_string(index=False))
    return df


def model_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Rank models by F1 and ROC-AUC for the final comparison table."""
    df = metrics_df.copy()
    df["score"] = (df["f1"] + df["roc_auc"]) / 2
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_importance(model: Any, X: np.ndarray,
                    feature_names: list[str]) -> tuple[pd.DataFrame, Any]:
    """Compute mean(|SHAP|) per feature using TreeExplainer when possible."""
    import shap

    LOG.info("Computing SHAP values ...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as exc:
        LOG.warning("TreeExplainer failed (%s) -- falling back to KernelExplainer.",
                    exc)
        background = shap.sample(X, 50, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X[:200])
        X = X[:200]

    if isinstance(shap_values, list):     # binary -> [class0, class1]
        sv = shap_values[1]
    else:
        sv = shap_values

    importance = np.abs(sv).mean(axis=0)
    df = (pd.DataFrame({"feature": feature_names,
                        "mean_abs_shap": importance})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))
    LOG.info("SHAP importance:\n%s", df.to_string(index=False))
    return df, sv


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_all(cluster_metrics: pd.DataFrame,
             cluster_summary: pd.DataFrame,
             clf_metrics: pd.DataFrame,
             comparison: pd.DataFrame,
             clf_results: dict,
             shap_df: pd.DataFrame | None) -> None:
    paths = get_paths()
    cluster_metrics.to_csv(paths["tables"] / "clustering_metrics.csv",
                           index=False)
    cluster_summary.to_csv(paths["tables"] / "cluster_summary.csv",
                           index=False)
    clf_metrics.to_csv(paths["tables"] / "classification_metrics.csv",
                       index=False)
    comparison.to_csv(paths["tables"] / "model_comparison.csv", index=False)

    for name, res in clf_results.items():
        report_df = pd.DataFrame(res.report).T.round(4)
        report_df.to_csv(
            paths["tables"] / f"classification_report_{name}.csv"
        )

    if shap_df is not None:
        shap_df.to_csv(paths["tables"] / "shap_feature_importance.csv",
                       index=False)

    LOG.info("All evaluation tables saved to %s", paths["tables"])
