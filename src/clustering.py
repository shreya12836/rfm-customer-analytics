"""
clustering.py
-------------
Unsupervised customer segmentation.

Models implemented
~~~~~~~~~~~~~~~~~~
- K-Means (with elbow + silhouette diagnostics for optimal k)
- DBSCAN
- Gaussian Mixture Model
- Deep clustering: PyTorch Autoencoder + KMeans on the latent space

Each clustering routine returns labels and a metric dict (silhouette,
Davies-Bouldin, Calinski-Harabasz). Noise points from DBSCAN are excluded
from internal validation indices since those metrics are undefined for
the noise label.

Business note
~~~~~~~~~~~~~
Segments derived here support strategic actions: champions to retain,
loyal customers to nurture, at-risk to win back, and one-time buyers to
either re-activate or de-prioritise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    from .utils import RANDOM_SEED, get_logger, get_paths, set_global_seed
except ImportError:
    from utils import RANDOM_SEED, get_logger, get_paths, set_global_seed  # type: ignore

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


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def kmeans_diagnostics(X: np.ndarray,
                       k_range: range = range(2, 11)) -> pd.DataFrame:
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
    """Pick the k that maximises silhouette score."""
    best = int(diag.loc[diag["silhouette"].idxmax(), "k"])
    LOG.info("Optimal k by silhouette = %d", best)
    return best


def fit_kmeans(X: np.ndarray, k: int) -> ClusteringResult:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    labels = km.fit_predict(X)
    metrics = _internal_metrics(X, labels)
    return ClusteringResult(
        name=f"KMeans(k={k})", labels=labels, metrics=metrics,
        extra={"model": km},
    )


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
def fit_dbscan(X: np.ndarray, eps: float = 0.5,
               min_samples: int = 10) -> ClusteringResult:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    metrics = _internal_metrics(X, labels)
    LOG.info("DBSCAN: %d clusters, %d noise points",
             metrics["n_clusters"], metrics["n_noise"])
    return ClusteringResult(name="DBSCAN", labels=labels, metrics=metrics,
                            extra={"model": db})


# ---------------------------------------------------------------------------
# Gaussian Mixture
# ---------------------------------------------------------------------------
def fit_gmm(X: np.ndarray, n_components: int) -> ClusteringResult:
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type="full",
                          random_state=RANDOM_SEED, n_init=3)
    labels = gmm.fit_predict(X)
    metrics = _internal_metrics(X, labels)
    metrics["bic"] = float(gmm.bic(X))
    metrics["aic"] = float(gmm.aic(X))
    return ClusteringResult(
        name=f"GMM(k={n_components})", labels=labels,
        metrics=metrics, extra={"model": gmm},
    )


# ---------------------------------------------------------------------------
# Autoencoder + KMeans (deep clustering)
# ---------------------------------------------------------------------------
def fit_autoencoder_kmeans(X: np.ndarray, k: int,
                           epochs: int = 50,
                           batch_size: int = 64,
                           lr: float = 1e-3) -> ClusteringResult:
    """Train an MLP autoencoder, KMeans-cluster the latent codes."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Autoencoder training on %s", device)

    in_dim = X.shape[1]

    class AE(nn.Module):
        """Input -> 16 -> 8 -> 4 (latent) -> 8 -> 16 -> Output."""

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
    metrics = _internal_metrics(latent_np, labels)
    return ClusteringResult(
        name=f"AE+KMeans(k={k})", labels=labels, metrics=metrics,
        extra={"history": history, "latent": latent_np, "model": model},
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_all_clustering(X: np.ndarray) -> dict[str, ClusteringResult]:
    """Run every clustering algorithm and return a {name: result} dict."""
    set_global_seed(RANDOM_SEED)

    diag = kmeans_diagnostics(X)
    k_opt = choose_optimal_k(diag)

    results = {
        "kmeans": fit_kmeans(X, k_opt),
        "dbscan": fit_dbscan(X, eps=0.6, min_samples=10),
        "gmm": fit_gmm(X, n_components=k_opt),
        "ae_kmeans": fit_autoencoder_kmeans(X, k=k_opt, epochs=50),
    }
    results["_diagnostics"] = ClusteringResult(
        name="kmeans_diag", labels=np.array([]),
        metrics={}, extra={"diag": diag, "k_opt": k_opt},
    )
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    X_demo = np.vstack([
        rng.normal(loc=(0, 0, 0), scale=0.4, size=(150, 3)),
        rng.normal(loc=(3, 3, 3), scale=0.4, size=(150, 3)),
        rng.normal(loc=(-3, 3, -3), scale=0.4, size=(150, 3)),
    ])
    res = run_all_clustering(X_demo)
    for name, r in res.items():
        if name.startswith("_"):
            continue
        LOG.info("%s -> %s", r.name, r.metrics)
