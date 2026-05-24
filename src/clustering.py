"""
clustering.py
-------------
Customer segmentation on the scaled RFM matrix.

Models:
- K-Means (k chosen by silhouette over cfg.modeling.k_range)
- DBSCAN
- Gaussian Mixture Model (k chosen by BIC over the same range)
- Autoencoder followed by KMeans on the latent codes

Each routine returns labels plus silhouette, Davies-Bouldin, and
Calinski-Harabasz on the non-noise subset. Which models run is set in
cfg.modeling.cluster_models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

try:
    from .config import PipelineConfig
    from .utils import RANDOM_SEED, get_logger, set_global_seed
except ImportError:
    from config import PipelineConfig  # type: ignore
    from utils import RANDOM_SEED, get_logger, set_global_seed  # type: ignore

LOG = get_logger("clustering")


@dataclass
class ClusteringResult:
    """Result bundle for a single clustering algorithm."""

    name: str
    labels: np.ndarray
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal evaluation
# ---------------------------------------------------------------------------
def _internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute silhouette, DB, CH on the labelled (non-noise) subset."""
    mask = labels != -1
    X_eval = X[mask]
    y_eval = labels[mask]
    n_clusters = len(set(y_eval))
    out = {"n_clusters": n_clusters, "n_noise": int((~mask).sum())}
    if n_clusters < 2 or len(X_eval) < n_clusters + 1:
        out.update(silhouette=np.nan, davies_bouldin=np.nan,
                   calinski_harabasz=np.nan)
        return out
    out["silhouette"] = float(silhouette_score(X_eval, y_eval))
    out["davies_bouldin"] = float(davies_bouldin_score(X_eval, y_eval))
    out["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, y_eval))
    return out


def _resolve_k_range(cfg: PipelineConfig) -> range:
    kr = cfg.modeling.k_range
    if len(kr) != 2:
        raise ValueError(f"k_range must be [start, stop], got {kr}")
    return range(kr[0], kr[1])


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def kmeans_diagnostics(X: np.ndarray, k_range: range) -> pd.DataFrame:
    """Compute inertia (elbow) and silhouette for each candidate k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": float(silhouette_score(X, labels)),
        })
    diag = pd.DataFrame(rows)
    LOG.info("K-Means diagnostics:\n%s", diag.to_string(index=False))
    return diag


def choose_optimal_k(diag: pd.DataFrame) -> int:
    best = int(diag.loc[diag["silhouette"].idxmax(), "k"])
    LOG.info("Optimal k (KMeans, silhouette) = %d", best)
    return best


def fit_kmeans(X: np.ndarray, k: int) -> ClusteringResult:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    labels = km.fit_predict(X)
    return ClusteringResult(
        name=f"KMeans(k={k})", labels=labels,
        metrics=_internal_metrics(X, labels), extra={"model": km},
    )


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
def fit_dbscan(X: np.ndarray, eps: float, min_samples: int) -> ClusteringResult:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    metrics = _internal_metrics(X, labels)
    LOG.info("DBSCAN(eps=%.2f, min_samples=%d): %d clusters, %d noise",
             eps, min_samples, metrics["n_clusters"], metrics["n_noise"])
    return ClusteringResult(name="DBSCAN", labels=labels, metrics=metrics,
                            extra={"model": db})


# ---------------------------------------------------------------------------
# Gaussian Mixture, k selected by BIC sweep
# ---------------------------------------------------------------------------
def fit_gmm_bic_sweep(X: np.ndarray, k_range: range) -> ClusteringResult:
    """Fit GMMs over k_range and return the one with the lowest BIC."""
    rows = []
    best_gmm, best_k, best_bic = None, None, np.inf
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=RANDOM_SEED, n_init=3)
        gmm.fit(X)
        bic = float(gmm.bic(X))
        rows.append({"k": k, "bic": bic, "aic": float(gmm.aic(X))})
        if bic < best_bic:
            best_bic, best_k, best_gmm = bic, k, gmm
    sweep = pd.DataFrame(rows)
    LOG.info("GMM BIC sweep:\n%s", sweep.to_string(index=False))
    LOG.info("Optimal k (GMM, BIC) = %d", best_k)

    labels = best_gmm.predict(X)
    metrics = _internal_metrics(X, labels)
    metrics.update(bic=best_bic, aic=float(best_gmm.aic(X)))
    return ClusteringResult(
        name=f"GMM(k={best_k})", labels=labels, metrics=metrics,
        extra={"model": best_gmm, "sweep": sweep, "k_opt": best_k},
    )


# ---------------------------------------------------------------------------
# Autoencoder + KMeans (deep clustering)
# ---------------------------------------------------------------------------
def fit_autoencoder_kmeans(X: np.ndarray, k: int,
                           epochs: int, batch_size: int,
                           lr: float) -> ClusteringResult:
    """Train an MLP autoencoder and run KMeans on the latent codes."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Autoencoder training on %s", device)

    in_dim = X.shape[1]

    class AE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 4), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(4, 8), nn.ReLU(),
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, in_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    model = AE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size,
                        shuffle=True)

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            running += loss.item() * batch.size(0)
        epoch_loss = running / len(tensor)
        history.append(epoch_loss)
        if epoch % 10 == 0 or epoch == 1:
            LOG.info("AE epoch %02d/%d  loss=%.5f", epoch, epochs, epoch_loss)

    model.eval()
    with torch.no_grad():
        _, latent = model(tensor.to(device))
        latent_np = latent.cpu().numpy()

    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    labels = km.fit_predict(latent_np)
    return ClusteringResult(
        name=f"AE+KMeans(k={k})", labels=labels,
        metrics=_internal_metrics(latent_np, labels),
        extra={"history": history, "latent": latent_np, "model": model},
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_all_clustering(X: np.ndarray,
                       cfg: PipelineConfig) -> dict[str, ClusteringResult]:
    """Run each algorithm listed in cfg.modeling.cluster_models."""
    set_global_seed(RANDOM_SEED)
    k_range = _resolve_k_range(cfg)

    diag = kmeans_diagnostics(X, k_range)
    k_opt = choose_optimal_k(diag)

    selected = set(cfg.modeling.cluster_models)
    results: dict[str, ClusteringResult] = {}

    if "kmeans" in selected:
        results["kmeans"] = fit_kmeans(X, k_opt)
    if "dbscan" in selected:
        results["dbscan"] = fit_dbscan(
            X, eps=cfg.modeling.dbscan_eps,
            min_samples=cfg.modeling.dbscan_min_samples,
        )
    if "gmm" in selected:
        results["gmm"] = fit_gmm_bic_sweep(X, k_range)
    if "ae_kmeans" in selected:
        ae_cfg = cfg.modeling.autoencoder
        results["ae_kmeans"] = fit_autoencoder_kmeans(
            X, k=k_opt, epochs=ae_cfg.epochs,
            batch_size=ae_cfg.batch_size, lr=ae_cfg.lr,
        )

    results["_diagnostics"] = ClusteringResult(
        name="kmeans_diag", labels=np.array([]),
        metrics={}, extra={"diag": diag, "k_opt": k_opt},
    )
    return results
