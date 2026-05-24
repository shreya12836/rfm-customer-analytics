"""
main.py
=======
Pipeline driver.

Usage:
    python main.py
    python main.py --config configs/online_retail_ii.yaml
    python main.py --config configs/my_dataset.yaml

All dataset, cleaning, target, and modeling choices are read from the
YAML config given by --config. To run on a new dataset, write a new
config file that maps the source schema to the canonical column names
used inside the pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import RANDOM_SEED, get_logger, get_paths, set_global_seed   # noqa: E402
from config import load_config                                          # noqa: E402
from preprocess import run_preprocessing                                # noqa: E402
from clustering import run_all_clustering                               # noqa: E402
from classification import run_classification                           # noqa: E402
import evaluation                                                       # noqa: E402
import visualisation as viz                                             # noqa: E402

LOG = get_logger("main")
DEFAULT_CONFIG = ROOT / "configs" / "online_retail_ii.yaml"


def banner(title: str) -> None:
    LOG.info("=" * 72)
    LOG.info(title)
    LOG.info("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RFM segmentation + classification pipeline")
    p.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG,
                   help=f"Path to YAML config (default: {DEFAULT_CONFIG})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(RANDOM_SEED)
    cfg = load_config(args.config)
    paths = get_paths()
    t0 = time.time()

    LOG.info("Running pipeline with config: %s (dataset=%s)",
             args.config, cfg.dataset.name)

    # ------------------------------------------------------------------
    banner("STEP 1 / 5  Data preprocessing & RFM construction")
    # ------------------------------------------------------------------
    art = run_preprocessing(cfg)
    art.rfm.to_csv(paths["tables"] / "rfm_table.csv", index=False)
    LOG.info("RFM table saved (n=%d customers).", len(art.rfm))

    # ------------------------------------------------------------------
    banner("STEP 2 / 5  Unsupervised clustering")
    # ------------------------------------------------------------------
    cluster_results = run_all_clustering(art.rfm_scaled, cfg)
    diag = cluster_results["_diagnostics"].extra["diag"]
    k_opt = cluster_results["_diagnostics"].extra["k_opt"]

    viz.plot_elbow(diag)
    viz.plot_silhouette(diag)
    if "ae_kmeans" in cluster_results:
        viz.plot_ae_loss(cluster_results["ae_kmeans"].extra["history"])
    if "kmeans" in cluster_results:
        viz.plot_rfm_clusters(art.rfm, cluster_results["kmeans"].labels)

    cluster_metrics = evaluation.clustering_metrics_table(cluster_results)
    cluster_summary = None
    if "kmeans" in cluster_results:
        cluster_summary = evaluation.cluster_summary_table(
            art.rfm, cluster_results["kmeans"].labels
        )

    # ------------------------------------------------------------------
    banner("STEP 3 / 5  Supervised classification (high-value customers)")
    # ------------------------------------------------------------------
    clf_bundle = run_classification(art.rfm, cfg)
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
        tree_candidates = [n for n in ("RandomForest", "XGBoost")
                           if n in clf_results]
        if tree_candidates:
            best_tree = max(tree_candidates,
                            key=lambda n: clf_results[n].metrics["f1"])
            LOG.info("Computing SHAP for %s ...", best_tree)
            shap_df, _ = evaluation.shap_importance(
                clf_results[best_tree].model,
                clf_bundle["X_test"],
                clf_bundle["feature_names"],
            )
            viz.plot_shap_importance(shap_df, best_tree)
        else:
            LOG.info("No tree classifier in config; skipping SHAP.")
    except Exception:                                # pragma: no cover
        LOG.warning("SHAP step failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    banner("STEP 5 / 5  Persist results")
    # ------------------------------------------------------------------
    evaluation.save_all(cluster_metrics, cluster_summary,
                        clf_metrics, comparison,
                        clf_results, shap_df)

    print("\n=========== CLUSTERING METRICS ===========")
    print(cluster_metrics.to_string(index=False))
    if cluster_summary is not None:
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
