"""
preprocess.py
-------------
Data ingestion and RFM feature engineering driven by a YAML config.

Steps:
1. Load source file (CSV / Excel / Parquet) per dataset.loader.
2. Rename source columns to canonical names per schema.
3. Apply cleaning rules per cleaning section.
4. Build per-customer Recency, Frequency, Monetary.
5. Optional log1p and scaler per features section.

Canonical column names after step 2:
    customer_id, transaction_id, transaction_date, quantity, unit_price, revenue

Downstream modules only reference canonical names.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    from .config import PipelineConfig
    from .utils import get_logger, get_paths
except ImportError:
    from config import PipelineConfig  # type: ignore
    from utils import get_logger, get_paths  # type: ignore

LOG = get_logger("preprocess")

# Output columns expected by clustering/classification:
RFM_COLS = ["Recency", "Frequency", "Monetary"]


@dataclass
class RFMArtifacts:
    """Container returned by run_preprocessing()."""

    raw: pd.DataFrame                 # cleaned transactions, canonical schema
    rfm: pd.DataFrame                 # one row per customer with R, F, M
    rfm_log: pd.DataFrame             # log1p-transformed RFM (if enabled)
    rfm_scaled: np.ndarray            # scaled matrix (n_customers, 3)
    scaler: object
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _resolve_path(p: str) -> Path:
    """Resolve dataset path relative to project root if not absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = get_paths()["root"] / path
    return path


def download_if_missing(target: Path, url: str | None) -> Path:
    """Download the dataset if it isn't on disk and a URL is configured."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000:
        LOG.info("Dataset already present at %s (%.1f MB)",
                 target, target.stat().st_size / 1e6)
        return target
    if url is None:
        raise FileNotFoundError(
            f"Dataset not found at {target} and no download_url set."
        )
    LOG.info("Downloading dataset from %s ...", url)
    urllib.request.urlretrieve(url, target)
    LOG.info("Downloaded %.1f MB to %s", target.stat().st_size / 1e6, target)
    return target


def load_raw(cfg: PipelineConfig) -> pd.DataFrame:
    """Load a source dataset based on dataset.loader and return a DataFrame."""
    path = _resolve_path(cfg.dataset.path)
    download_if_missing(path, cfg.dataset.download_url)

    loader = cfg.dataset.loader
    LOG.info("Loading %s via %s loader ...", path, loader)

    if loader == "excel":
        sheets = cfg.dataset.excel_sheets
        if sheets == "all":
            data = pd.read_excel(path, sheet_name=None, engine="openpyxl")
            df = pd.concat(data.values(), ignore_index=True)
        else:
            df = pd.read_excel(path, sheet_name=sheets, engine="openpyxl")
            if isinstance(df, dict):
                df = pd.concat(df.values(), ignore_index=True)
    elif loader == "csv":
        df = pd.read_csv(path)
    elif loader == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported loader: {loader}")

    df.columns = [c.strip() for c in df.columns]
    LOG.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Schema mapping
# ---------------------------------------------------------------------------
def map_schema(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Rename source columns to canonical names; compute revenue if needed."""
    s = cfg.schema
    rename = {
        s.customer_id: "customer_id",
        s.transaction_id: "transaction_id",
        s.transaction_date: "transaction_date",
    }
    if s.quantity:
        rename[s.quantity] = "quantity"
    if s.unit_price:
        rename[s.unit_price] = "unit_price"
    if s.revenue:
        rename[s.revenue] = "revenue"

    missing = [src for src in rename if src not in df.columns]
    if missing:
        raise ValueError(
            f"Source columns missing from data: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.rename(columns=rename)

    # Compute revenue if not explicitly mapped
    if "revenue" not in df.columns:
        df["revenue"] = df["quantity"] * df["unit_price"]

    keep = [c for c in cfg.canonical_columns() if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_transactions(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Apply cleaning rules from cfg.cleaning."""
    n0 = len(df)
    c = cfg.cleaning

    if c.drop_cancellations.enabled:
        method = c.drop_cancellations.method
        if method == "invoice_prefix":
            prefix = c.drop_cancellations.prefix
            df = df[~df["transaction_id"].astype(str).str.startswith(prefix)]
        elif method == "negative_quantity":
            if "quantity" in df.columns:
                df = df[df["quantity"] >= 0]
        LOG.info("After cancellation drop (%s): %d rows", method, len(df))

    if c.drop_missing_customer:
        df = df.dropna(subset=["customer_id"])
        try:
            df["customer_id"] = df["customer_id"].astype(int)
        except (ValueError, TypeError):
            df["customer_id"] = df["customer_id"].astype(str)

    if c.drop_nonpositive:
        if "quantity" in df.columns:
            df = df[df["quantity"] > 0]
        if "unit_price" in df.columns:
            df = df[df["unit_price"] > 0]
        df = df[df["revenue"] > 0]

    if c.drop_duplicates:
        df = df.drop_duplicates()

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    LOG.info("Cleaning: %d -> %d rows (%.1f%% retained)",
             n0, len(df), 100 * len(df) / max(n0, 1))
    return df.reset_index(drop=True)


def remove_outliers(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Apply outlier removal as configured."""
    o = cfg.cleaning.outliers
    if o.method == "none":
        return df

    cols = [c for c in o.columns if c in df.columns]
    if not cols:
        LOG.info("Outlier removal skipped: none of %s present", o.columns)
        return df

    mask = pd.Series(True, index=df.index)
    if o.method == "iqr":
        for col in cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - o.k * iqr, q3 + o.k * iqr
            mask &= df[col].between(lo, hi)
            LOG.info("  %s in [%.3f, %.3f]", col, lo, hi)
    elif o.method == "zscore":
        for col in cols:
            mu, sigma = df[col].mean(), df[col].std()
            z = (df[col] - mu) / sigma
            mask &= z.abs() <= o.k
            LOG.info("  %s |z| <= %.1f", col, o.k)

    out = df.loc[mask].reset_index(drop=True)
    LOG.info("Outlier removal (%s, k=%.1f): %d -> %d rows",
             o.method, o.k, len(df), len(out))
    return out


# ---------------------------------------------------------------------------
# RFM feature engineering
# ---------------------------------------------------------------------------
def build_rfm(df: pd.DataFrame, cfg: PipelineConfig) -> RFMArtifacts:
    """Aggregate transactions into a customer-level RFM table."""
    LOG.info("Building RFM features ...")

    snap_cfg = cfg.features.snapshot_date
    if snap_cfg == "auto":
        snapshot = df["transaction_date"].max() + pd.Timedelta(days=1)
    else:
        snapshot = pd.to_datetime(snap_cfg)
    LOG.info("Snapshot date: %s", snapshot.date())

    rfm = (
        df.groupby("customer_id")
        .agg(
            Recency=("transaction_date", lambda s: (snapshot - s.max()).days),
            Frequency=("transaction_id", "nunique"),
            Monetary=("revenue", "sum"),
        )
        .reset_index()
        .rename(columns={"customer_id": "CustomerID"})
    )

    rfm["Monetary"] = rfm["Monetary"].clip(lower=0.01)
    rfm["Frequency"] = rfm["Frequency"].clip(lower=1)
    rfm["Recency"] = rfm["Recency"].clip(lower=0)

    rfm_log = rfm.copy()
    if cfg.features.log_transform:
        rfm_log[RFM_COLS] = np.log1p(rfm_log[RFM_COLS])

    scaler = _make_scaler(cfg.features.scaler)
    if scaler is None:
        scaled = rfm_log[RFM_COLS].values
    else:
        scaled = scaler.fit_transform(rfm_log[RFM_COLS].values)

    LOG.info("RFM table: %d customers", len(rfm))
    return RFMArtifacts(
        raw=df, rfm=rfm, rfm_log=rfm_log, rfm_scaled=scaled,
        scaler=scaler, feature_names=RFM_COLS,
    )


def _make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "none":
        return None
    raise ValueError(f"Unsupported scaler: {kind}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_preprocessing(cfg: PipelineConfig) -> RFMArtifacts:
    """Run load, schema mapping, cleaning, outlier removal, and RFM build."""
    raw = load_raw(cfg)
    mapped = map_schema(raw, cfg)
    cleaned = clean_transactions(mapped, cfg)
    no_outliers = remove_outliers(cleaned, cfg)
    return build_rfm(no_outliers, cfg)


if __name__ == "__main__":
    from config import load_config
    cfg = load_config(get_paths()["root"] / "configs" / "online_retail_ii.yaml")
    art = run_preprocessing(cfg)
    print(art.rfm.describe())
