"""
visualisation.py
----------------
Publication-quality figures (>= 250 dpi PNG) saved to outputs/figures/:

- elbow.png                : K-Means inertia vs. k (elbow curve)
- silhouette.png           : silhouette score vs. k
- ae_loss.png              : autoencoder reconstruction loss curve
- rfm_clusters_3d.png      : 3-D RFM scatter coloured by cluster
- roc_curves.png           : ROC curves for all classifiers
- shap_importance.png      : mean |SHAP| bar plot
- confusion_<model>.png    : confusion matrix per classifier
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D)

try:
    from .utils import get_logger, get_paths
except ImportError:
    from utils import get_logger, get_paths  # type: ignore

LOG = get_logger("visualisation")
DPI = 300                                   # > 250 dpi required by spec
sns.set_theme(style="whitegrid", context="paper")


def _save(fig: plt.Figure, name: str) -> Path:
    paths = get_paths()
    out = paths["figures"] / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", out)
    return out


# ---------------------------------------------------------------------------
# Clustering diagnostics
# ---------------------------------------------------------------------------
def plot_elbow(diag: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(diag["k"], diag["inertia"], "o-", color="steelblue")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("Elbow Method for Optimal k")
    _save(fig, "elbow.png")


def plot_silhouette(diag: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(diag["k"], diag["silhouette"], "s-", color="darkorange")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette Analysis for K-Means")
    best = diag.loc[diag["silhouette"].idxmax()]
    ax.axvline(best["k"], ls="--", color="grey", alpha=0.7,
               label=f"k* = {int(best['k'])}")
    ax.legend()
    _save(fig, "silhouette.png")


# ---------------------------------------------------------------------------
# Autoencoder loss
# ---------------------------------------------------------------------------
def plot_ae_loss(history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history, color="seagreen")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE reconstruction loss")
    ax.set_title("Autoencoder Training Loss")
    _save(fig, "ae_loss.png")


# ---------------------------------------------------------------------------
# 3-D RFM cluster scatter
# ---------------------------------------------------------------------------
def plot_rfm_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        rfm["Recency"], rfm["Frequency"], rfm["Monetary"],
        c=labels, cmap="tab10", s=12, alpha=0.7,
    )
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Frequency (orders)")
    ax.set_zlabel("Monetary (GBP)")
    ax.set_title("3-D RFM Customer Segmentation")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster",
                       loc="upper left", fontsize=8)
    ax.add_artist(legend)
    _save(fig, "rfm_clusters_3d.png")


# ---------------------------------------------------------------------------
# Classification visualisations
# ---------------------------------------------------------------------------
def plot_roc_curves(clf_results: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, res in clf_results.items():
        fpr, tpr, _ = res.roc_curve
        ax.plot(fpr, tpr, label=f"{name} (AUC={res.metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- High-Value Customer Classification")
    ax.legend()
    _save(fig, "roc_curves.png")


def plot_confusion(cm: np.ndarray, name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Low", "High"],
                yticklabels=["Low", "High"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {name}")
    _save(fig, f"confusion_{name}.png")


# ---------------------------------------------------------------------------
# SHAP importance
# ---------------------------------------------------------------------------
def plot_shap_importance(shap_df: pd.DataFrame, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=shap_df, x="mean_abs_shap", y="feature",
                color="indigo", ax=ax)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    ax.set_title(f"Feature Importance via SHAP -- {model_name}")
    _save(fig, "shap_importance.png")
