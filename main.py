"""
main.py
=======
End-to-end driver for the project:

    Customer Segmentation and Purchase Prediction using
    Statistical Machine Learning Techniques

Author      : Shreya Mishra (IED/10032/23)
Branch      : Centre of Quantitative Economics and Data Science
Course      : ED317 - Statistical Machine Learning I
Institution : Birla Institute of Technology, Mesra, Ranchi
Submitted to: Dr. Manish Kumar Pandey
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import RANDOM_SEED, get_logger, get_paths, set_global_seed   # noqa: E402
from preprocess import run_preprocessing                                # noqa: E402
from clustering import run_all_clustering                               # noqa: E402
from classification import run_classification                           # noqa: E402
import evaluation                                                       # noqa: E402
import visualisation as viz                                             # noqa: E402

LOG = get_logger("main")


def banner(title: str) -> None:
    LOG.info("=" * 72)
    LOG.info(title)
    LOG.info("=" * 72)


def main() -> None:
    set_global_seed(RANDOM_SEED)
    paths = get_paths()
    t0 = time.time()

    # ------------------------------------------------------------------
    banner("STEP 1 / 5  Data preprocessing & RFM construction")
    # ------------------------------------------------------------------
    art = run_preprocessing()
    art.rfm.to_csv(paths["tables"] / "rfm_table.csv", index=False)
    LOG.info("RFM table saved (n=%d customers).", len(art.rfm))

    # ------------------------------------------------------------------
    banner("STEP 2 / 5  Unsupervised clustering")
    # ------------------------------------------------------------------
    cluster_results = run_all_clustering(art.rfm_scaled)
    diag = cluster_results["_diagnostics"].extra["diag"]
    k_opt = cluster_results["_diagnostics"].extra["k_opt"]

    # Diagnostics + AE loss + 3-D scatter
    viz.plot_elbow(diag)
    viz.plot_silhouette(diag)
    viz.plot_ae_loss(cluster_results["ae_kmeans"].extra["history"])
    viz.plot_rfm_clusters(art.rfm, cluster_results["kmeans"].labels)

    cluster_metrics = evaluation.clustering_metrics_table(cluster_results)
    cluster_summary = evaluation.cluster_summary_table(
        art.rfm, cluster_results["kmeans"].labels
    )

    # ------------------------------------------------------------------
    banner("STEP 3 / 5  Supervised classification (high-value customers)")
    # ------------------------------------------------------------------
    clf_bundle = run_classification(art.rfm)
    clf_results = clf_bundle["results"]

    clf_metrics = evaluation.classification_metrics_table(clf_results)
    comparison = evaluation.model_comparison(clf_metrics)

    viz.plot_roc_curves(clf_results)
    for name, res in clf_results.items():
        viz.plot_confusion(res.confusion, name)

    # ------------------------------------------------------------------
    banner("STEP 4 / 5  SHAP explainability")
    # ------------------------------------------------------------------
    shap_df = None
    try:
        # Pick the better tree model (RF or XGBoost) by F1
        best_tree = max(("RandomForest", "XGBoost"),
                        key=lambda n: clf_results[n].metrics["f1"])
        LOG.info("Computing SHAP for %s ...", best_tree)
        shap_df, _ = evaluation.shap_importance(
            clf_results[best_tree].model,
            clf_bundle["X_test"],
            clf_bundle["feature_names"],
        )
        viz.plot_shap_importance(shap_df, best_tree)
    except Exception:                                # pragma: no cover
        LOG.warning("SHAP step failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    banner("STEP 5 / 5  Persist results")
    # ------------------------------------------------------------------
    evaluation.save_all(cluster_metrics, cluster_summary,
                        clf_metrics, comparison,
                        clf_results, shap_df)

    # Console summary
    print("\n=========== CLUSTERING METRICS ===========")
    print(cluster_metrics.to_string(index=False))
    print("\n=========== CLUSTER SUMMARY ===========")
    print(cluster_summary.to_string(index=False))
    print("\n=========== CLASSIFICATION METRICS ===========")
    print(clf_metrics.to_string(index=False))
    print("\n=========== FINAL MODEL COMPARISON ===========")
    print(comparison.to_string(index=False))
    if shap_df is not None:
        print("\n=========== SHAP IMPORTANCE ===========")
        print(shap_df.to_string(index=False))

    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.1f} s")
    LOG.info("Optimal k (KMeans) = %d", k_opt)
    LOG.info("Figures : %s", paths["figures"])
    LOG.info("Tables  : %s", paths["tables"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.error("Pipeline failed:\n%s", traceback.format_exc())
        sys.exit(1)
