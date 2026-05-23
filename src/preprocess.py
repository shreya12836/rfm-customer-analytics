"""
preprocess.py
-------------
Data ingestion and preprocessing for the UCI Online Retail II dataset.

Pipeline
~~~~~~~~
1. Load Excel (auto-download if absent).
2. Clean transactions (cancellations, missing CustomerID, non-positive
   quantities/prices, duplicates).
3. Remove outliers using the IQR rule on Quantity and UnitPrice.
4. Engineer RFM features (Recency, Frequency, Monetary) per customer.
5. Apply log1p transform and StandardScaler for downstream modelling.

Business note
~~~~~~~~~~~~~
RFM is the canonical framework in retail analytics: it captures *when* a
customer last bought (Recency), *how often* they buy (Frequency) and *how
much* they spend (Monetary). These three signals jointly characterise
customer value and form the basis of segmentation and revenue prediction.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .utils import RANDOM_SEED, get_logger, get_paths
except ImportError:
    from utils import RANDOM_SEED, get_logger, get_paths  # type: ignore

LOG = get_logger("preprocess")

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/"
    "online_retail_II.xlsx"
)
LOCAL_FILENAME = "online_retail_II.xlsx"


@dataclass
class RFMArtifacts:
    """Container returned by build_rfm() with everything downstream needs."""

    raw: pd.DataFrame                 # cleaned transactions
    rfm: pd.DataFrame                 # one row per CustomerID with R,F,M
    rfm_log: pd.DataFrame             # log1p-transformed R,F,M
    rfm_scaled: np.ndarray            # standardized matrix (n_customers, 3)
    scaler: StandardScaler
    feature_names: list


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def download_dataset(target: Path) -> Path:
    """Download the UCI Online Retail II Excel file if it is not on disk."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000_000:
        LOG.info("Dataset already present at %s (%.1f MB)",
                 target, target.stat().st_size / 1e6)
        return target

    LOG.info("Downloading UCI Online Retail II from %s ...", UCI_URL)
    try:
        urllib.request.urlretrieve(UCI_URL, target)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"Failed to download dataset from {UCI_URL}: {exc}"
        ) from exc
    LOG.info("Downloaded %.1f MB to %s", target.stat().st_size / 1e6, target)
    return target


def load_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the Online Retail II workbook and concatenate both sheets."""
    paths = get_paths()
    data_dir = data_dir or paths["data"]
    xlsx_path = data_dir / LOCAL_FILENAME
    download_dataset(xlsx_path)

    LOG.info("Reading Excel workbook (this can take a minute) ...")
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    LOG.info("Loaded %d raw transactions across %d sheets",
             len(df), len(sheets))
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancellations, missing customers, non-positive values, duplicates."""
    LOG.info("Cleaning transactions ...")
    n0 = len(df)

    # The two sheets use different casing; normalise.
    rename_map = {"Customer ID": "CustomerID"}
    df = df.rename(columns=rename_map)

    required = {"Invoice", "StockCode", "Quantity", "InvoiceDate",
                "Price", "CustomerID", "Country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns missing from dataset: {missing}")

    # Remove cancellations (Invoice numbers starting with 'C')
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Keep only positive quantities and prices
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Ensure datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Total line revenue
    df["Revenue"] = df["Quantity"] * df["Price"]

    LOG.info("Cleaned: %d -> %d rows (%.1f%% retained)",
             n0, len(df), 100 * len(df) / n0)
    return df.reset_index(drop=True)


def remove_outliers_iqr(df: pd.DataFrame,
                        cols: tuple[str, ...] = ("Quantity", "Price"),
                        k: float = 1.5) -> pd.DataFrame:
    """Remove rows whose Quantity/Price fall outside [Q1 - k*IQR, Q3 + k*IQR]."""
    LOG.info("Removing outliers via IQR (k=%.1f) on %s", k, list(cols))
    mask = pd.Series(True, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        before = mask.sum()
        mask &= df[col].between(lo, hi)
        LOG.info("  %s in [%.3f, %.3f] -> kept %d (%d dropped)",
                 col, lo, hi, mask.sum(), before - mask.sum())
    out = df.loc[mask].reset_index(drop=True)
    LOG.info("Outlier removal: %d -> %d rows", len(df), len(out))
    return out


# ---------------------------------------------------------------------------
# RFM feature engineering
# ---------------------------------------------------------------------------
def build_rfm(df: pd.DataFrame) -> RFMArtifacts:
    """Aggregate transactions to a customer-level RFM table."""
    LOG.info("Building RFM features ...")
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )
    # Guard against zeros before log
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0.01)
    rfm["Frequency"] = rfm["Frequency"].clip(lower=1)
    rfm["Recency"] = rfm["Recency"].clip(lower=0)

    LOG.info("RFM table: %d customers", len(rfm))

    log_cols = ["Recency", "Frequency", "Monetary"]
    rfm_log = rfm.copy()
    rfm_log[log_cols] = np.log1p(rfm_log[log_cols])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_log[log_cols].values)

    return RFMArtifacts(
        raw=df,
        rfm=rfm,
        rfm_log=rfm_log,
        rfm_scaled=scaled,
        scaler=scaler,
        feature_names=log_cols,
    )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def run_preprocessing() -> RFMArtifacts:
    """Convenience entry point: load -> clean -> outliers -> RFM."""
    raw = load_raw()
    cleaned = clean_transactions(raw)
    no_outliers = remove_outliers_iqr(cleaned)
    return build_rfm(no_outliers)


if __name__ == "__main__":
    art = run_preprocessing()
    print(art.rfm.describe())
    paths = get_paths()
    art.rfm.to_csv(paths["tables"] / "rfm_table.csv", index=False)
    LOG.info("Saved RFM table to %s", paths["tables"] / "rfm_table.csv")
